import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
class WrapperCodeGen(CodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def __init__(self):
        super().__init__()
        self._names_iter = count()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.src_to_kernel = {}
        self.kenel_numel_expr = set()
        self.lines = []
        self.declare = ''
        self.ending = ''
        self.open_bracket = '['
        self.closed_bracket = ']'
        self.comment = '#'
        self.namespace = ''
        self.none_str = 'None'
        self.size = 'size()'
        self.stride = 'stride()'
        self.last_seen_device_guard_index = None
        self.supports_intermediate_hooks = True
        self.expr_printer = pexpr
        self.cached_thread_locals = set()
        self.user_defined_kernel_cache: Dict[Tuple[Any, ...], str] = {}
        self.unbacked_symbol_decls = set()
        self.write_header()
        self.write_prefix()
        if not V.graph.aot_mode:
            for name, hashed in V.graph.constant_reprs.items():
                self.write_constant(name, hashed)
        self.allocated = set()
        self.freed: Set[str] = set()
        self.reuses = dict()
        self.write_get_raw_stream = functools.lru_cache(None)(self.write_get_raw_stream)

        @functools.lru_cache(None)
        def add_import_once(line):
            self.header.writeline(line)
        self.add_import_once = add_import_once
        self._metas = {}

    def write_constant(self, name, hashed):
        self.header.writeline(f'{name} = None  # {hashed}')

    def write_header(self):
        self.header.splice(f'\n                from ctypes import c_void_p, c_long\n                import torch\n                import math\n                import random\n                import os\n                import tempfile\n                from math import inf, nan\n                from torch._inductor.hooks import run_intermediate_hooks\n                from torch._inductor.utils import maybe_profile\n                from torch._inductor.codegen.memory_planning import _align as align\n\n                from torch import device, empty, empty_strided\n                from {codecache.__name__} import AsyncCompile\n                from torch._inductor.select_algorithm import extern_kernels\n\n                aten = torch.ops.aten\n                inductor_ops = torch.ops.inductor\n                assert_size_stride = torch._C._dynamo.guards.assert_size_stride\n                alloc_from_pool = torch.ops.inductor._alloc_from_pool\n                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor\n                async_compile = AsyncCompile()\n\n            ')

    @cache_on_self
    def write_triton_header_once(self):
        self.header.splice('\n            import triton\n            import triton.language as tl\n            from torch._inductor.triton_heuristics import grid, start_graph, end_graph\n            from torch._C import _cuda_getCurrentRawStream as get_cuda_stream\n            ')

    def add_meta_once(self, meta):
        meta = repr(meta)
        if meta not in self._metas:
            var = f'meta{len(self._metas)}'
            self._metas[meta] = var
            self.header.writeline(f'{var} = {meta}')
        return self._metas[meta]

    @cache_on_self
    def get_output_refs(self):
        return [x.codegen_reference(self.wrapper_call) for x in V.graph.graph_outputs]

    def mark_output_type(self):
        return

    def codegen_input_size_asserts(self):
        for name, buf in V.graph.graph_inputs.items():
            if isinstance(buf, sympy.Expr):
                continue
            if sympy_product(buf.get_size()) == 0:
                continue
            size = self.codegen_shape_tuple(buf.get_size())
            stride = self.codegen_shape_tuple(buf.get_stride())
            self.prefix.writeline(f'assert_size_stride({name}, {size}, {stride})')

    def write_prefix(self):
        self.prefix.splice('\n\n            async_compile.wait(globals())\n            del async_compile\n\n            def call(args):\n            ')
        with self.prefix.indent():
            if config.triton.debug_sync_graph:
                self.prefix.writeline('torch.cuda.synchronize()')
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f'{', '.join(V.graph.graph_inputs.keys())}{('' if inp_len != 1 else ',')}'
                self.prefix.writeline(f'{lhs} = args')
                self.prefix.writeline('args.clear()')
            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            if config.size_asserts:
                self.codegen_input_size_asserts()

    def write_get_raw_stream(self, index):
        self.write_triton_header_once()
        name = f'stream{index}'
        self.writeline(f'{name} = get_cuda_stream({index})')
        return name

    def next_kernel_suffix(self):
        return f'{next(self._names_iter)}'

    def codegen_device_guard_enter(self, device_idx):
        self.writeline(EnterCudaDeviceContextManagerLine(device_idx, self.last_seen_device_guard_index))
        self.last_seen_device_guard_index = device_idx

    def codegen_device_guard_exit(self):
        self.writeline(ExitCudaDeviceContextManagerLine())

    def generate_return(self, output_refs):
        if output_refs:
            self.wrapper_call.writeline('return (' + ', '.join(output_refs) + ', )')
        else:
            self.wrapper_call.writeline('return ()')

    def generate_end(self, result):
        return

    def generate_fallback_kernel(self, fallback_kernel, args):
        self.generate_extern_kernel_alloc(fallback_kernel, args)

    def generate_extern_kernel_alloc(self, extern_kernel, args):
        ending = self.ending
        if config.memory_planning and 'view_as_complex' in str(extern_kernel.kernel):
            ending = f'.clone(){ending}'
        output_name = extern_kernel.get_name()
        origin_node = extern_kernel.get_origin_node()
        kernel_name = extern_kernel.codegen_kernel_name()
        self.writeline(f'{self.declare}{output_name} = {kernel_name}({', '.join(args)}){ending}')
        if self.supports_intermediate_hooks and config.generate_intermediate_hooks and (origin_node is not None):
            counters['inductor']['intermediate_hooks'] += 1
            self.writeline(f'run_intermediate_hooks({origin_node.name!r}, {output_name})')

    def generate_extern_kernel_out(self, output_view, codegen_reference, args, kernel):
        if output_view:
            args.append(f'out={output_view.codegen_reference()}')
        else:
            args.append(f'out={codegen_reference}')
        self.writeline(f'{kernel}({', '.join(args)})')

    def generate_user_defined_triton_kernel(self, kernel_name, grid, configs, args):
        grid, code = user_defined_kernel_grid_fn_code(kernel_name, configs, grid)
        with self.prefix.indent():
            self.prefix.splice(code)
        stream_name = self.write_get_raw_stream(V.graph.scheduler.current_device.index)
        self.writeline(f'{kernel_name}.run({', '.join(args)}, grid={grid}, stream={stream_name})')

    def generate_scatter_fallback(self, output, inputs, kernel, fn, src_is_tensor, reduce, kwargs):
        line = f'{kernel}({','.join(map(str, inputs))}'
        if kernel == 'aten.scatter_':
            if reduce:
                line += f', reduce={repr(reduce)}'
        else:
            line += ', '.join([''] + kwargs)
        line += f'){self.ending}'
        self.writeline(line)

    def generate_extern_kernel_alloc_and_find_schema_if_needed(self, name, kernel, codegen_args, cpp_op_schema, cpp_kernel_key, cpp_kernel_overload_name='', op_overload=None, raw_args=None, outputs=None):
        self.writeline(f'{name} = {kernel}({', '.join(codegen_args)})')

    def generate_inf_and_nan_checker(self, node):
        pass

    @dynamo_timed
    def generate(self, is_inference):
        if config.profile_bandwidth:
            self.write_triton_header_once()
        result = IndentedBuffer()
        result.splice(self.header)
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.generate_start_graph()
            if is_inference and config.memory_planning:
                self.memory_plan()
            else:
                self.memory_plan_reuse()
            device_cm_stack = contextlib.ExitStack()
            for line in self.lines:
                if isinstance(line, MemoryPlanningLine):
                    line.codegen(self.wrapper_call)
                elif isinstance(line, (EnterCudaDeviceContextManagerLine, ExitCudaDeviceContextManagerLine)):
                    line.codegen(self.wrapper_call, device_cm_stack)
                else:
                    self.wrapper_call.writeline(line)
            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline('torch.cuda.synchronize()')
            if config.profile_bandwidth:
                self.generate_end_graph()
            self.generate_return(output_refs)
        self.append_precomputed_sizes_to_prefix()
        self.finalize_prefix()
        result.splice(self.prefix)
        with result.indent():
            result.splice(self.wrapper_call)
        self.generate_end(result)
        self.add_benchmark_harness(result)
        return result.getvaluewithlinemap()

    def memory_plan(self):
        from .memory_planning import MemoryPlanner
        self.lines = MemoryPlanner(self).plan(self.lines)

    def memory_plan_reuse(self):
        out_names = V.graph.get_output_names()
        while self.lines and isinstance(self.lines[-1], MemoryPlanningLine) and (self.lines[-1].node.name not in out_names):
            self.lines.pop()
        planning_state = MemoryPlanningState()
        for i in range(len(self.lines)):
            if isinstance(self.lines[i], MemoryPlanningLine):
                self.lines[i] = self.lines[i].plan(planning_state)

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f'{self.declare}{name}_size = {name}.{self.size}{self.ending}')

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f'{self.declare}{name}_stride = {name}.{self.stride}{self.ending}')

    def codegen_inputs(self, code: IndentedBuffer, graph_inputs: Dict[str, ir.TensorBox]):
        """Assign all symbolic shapes to locals"""

        @functools.lru_cache(None)
        def sizeof(name):
            self.codegen_input_size_var_decl(code, name)
            return f'{name}_size'

        @functools.lru_cache(None)
        def strideof(name):
            self.codegen_input_stride_var_decl(code, name)
            return f'{name}_stride'
        needed = V.graph.sizevars.free_symbols()

        def is_expr(x):
            return isinstance(x[1], sympy.Expr)
        graph_inputs_expr = list(filter(is_expr, graph_inputs.items()))
        graph_inputs_tensors = list(filter(lambda x: not is_expr(x), graph_inputs.items()))
        for name, shape in graph_inputs_expr:
            shape = V.graph.sizevars.simplify(shape)
            if shape in needed:
                needed.remove(shape)
                code.writeline(f'{self.declare}{shape} = {name}{self.ending}')
        for name, value in graph_inputs_tensors:
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(f'{self.declare}{shape} = {sizeof(name)}[{dim}]{self.ending}')
        for name, value in graph_inputs_tensors:
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                shape = V.graph.sizevars.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(f'{self.declare}{shape} = {strideof(name)}[{dim}]{self.ending}')

    def append_precomputed_sizes_to_prefix(self):
        with self.prefix.indent():
            for sym, expr in V.graph.sizevars.inv_precomputed_replacements.items():
                self.prefix.writeline(f'{self.declare}{sym} = {self.expr_printer(expr)}{self.ending}')

    def finalize_prefix(self):
        pass

    def codegen_python_sizevar(self, x: Expr) -> str:
        return pexpr(V.graph.sizevars.simplify(x))

    def codegen_sizevar(self, x: Expr) -> str:
        return self.codegen_python_sizevar(x)

    def codegen_tuple_access(self, basename: str, name: str, index: str) -> str:
        return f'{basename}[{index}]'

    def codegen_python_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_python_sizevar, shape))
        if len(parts) == 0:
            return '()'
        if len(parts) == 1:
            return f'({parts[0]}, )'
        return f'({', '.join(parts)})'

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        return self.codegen_python_shape_tuple(shape)

    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        return 'alloc_from_pool({})'.format(', '.join([name, pexpr(offset), str(dtype), self.codegen_shape_tuple(shape), self.codegen_shape_tuple(stride)]))

    def codegen_reinterpret_view(self, data, size, stride, offset, writer) -> str:
        size = self.codegen_shape_tuple(size)
        stride = self.codegen_shape_tuple(stride)
        offset = self.codegen_sizevar(offset)
        return f'reinterpret_tensor({data.get_name()}, {size}, {stride}, {offset})'

    def codegen_device_copy(self, src, dst):
        self.writeline(f'{dst}.copy_({src})')

    def codegen_multi_output(self, name, value):
        self.writeline(f'{self.declare}{name} = {value}{self.ending}')

    def benchmark_compiled_module(self, output):

        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(f"{name} = rand_strided({self.codegen_python_shape_tuple(shape)}, {self.codegen_python_shape_tuple(stride)}, device='{device}', dtype={dtype})")

        def add_expr_input(name, val):
            output.writeline(f'{name} = {val}')
        output.writelines(['', '', 'def benchmark_compiled_module(times=10, repeat=10):'])
        with output.indent():
            output.splice('\n                from torch._dynamo.testing import rand_strided\n                from torch._inductor.utils import print_performance\n                ', strip=True)
            for name, value in V.graph.constants.items():
                output.writeline(f'global {name}')
                add_fake_input(name, value.size(), value.stride(), value.device, value.dtype)
            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(V.graph.sizevars.var_to_val.get(value, None), SingletonInt):
                    continue
                if isinstance(value, sympy.Expr):
                    add_expr_input(name, V.graph.sizevars.size_hint(value))
                else:
                    shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                    stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                    add_fake_input(name, shape, stride, value.get_device(), value.get_dtype())
            call_str = f'call([{', '.join(V.graph.graph_inputs.keys())}])'
            output.writeline(f'fn = lambda: {call_str}')
            output.writeline('return print_performance(fn, times=times, repeat=repeat)')

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return
        self.benchmark_compiled_module(output)
        output.writelines(['', '', 'if __name__ == "__main__":'])
        with output.indent():
            output.writelines(['from torch._inductor.wrapper_benchmark import compiled_module_main', f"compiled_module_main('{get_benchmark_name()}', benchmark_compiled_module)"])

    def define_kernel(self, name: str, kernel: str, metadata: Optional[str]=None, cuda=True):
        metadata_comment = f'{metadata}\n' if metadata else ''
        self.header.splice(f'\n\n{metadata_comment}{name} = {kernel}')

    def define_user_defined_triton_kernel(self, kernel, configs, kwargs):
        original_name = kernel.__name__
        cache_key = [id(kernel.fn)]
        for arg in kwargs.values():
            if isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                cache_key.append(arg.get_dtype())
            elif len(configs) > 0:
                cache_key.append(arg)
        cache_key = tuple(cache_key)
        if cache_key in self.user_defined_kernel_cache:
            return self.user_defined_kernel_cache[cache_key]
        name = f'{original_name}_{len(self.user_defined_kernel_cache)}'
        self.user_defined_kernel_cache[cache_key] = name
        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.triton({original_name!r}, '''")
        compile_wrapper.splice('\n            import triton\n            import triton.language as tl\n            from torch._inductor.utils import instance_descriptor\n            from torch._inductor.triton_heuristics import user_autotune\n            ', strip=True)
        compile_wrapper.newline()
        from .common import SizeArg, TensorArg
        signature: List[Union[TensorArg, SizeArg]] = []
        constants = {}
        for key, arg in kwargs.items():
            idx = kernel.arg_names.index(key)
            if idx in kernel.constexprs:
                constants[key] = arg
                continue
            if isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
                signature.append(TensorArg(key, arg.codegen_reference(), arg.get_dtype(), not isinstance(arg, ReinterpretView)))
            else:
                signature.append(SizeArg(key, arg))
        index_dtype = 'tl.int32'
        inductor_meta = {'kernel_name': name}
        triton_meta = {'signature': signature_to_meta(signature, size_dtype=index_dtype), 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': constants, 'configs': [config_of(signature)]}
        configs = [{'kwargs': config.kwargs, 'num_warps': config.num_warps, 'num_stages': config.num_stages} for config in configs]
        compile_wrapper.splice(f'\n            @user_autotune(\n                configs={configs!r},\n                inductor_meta={inductor_meta!r},\n                triton_meta={triton_meta!r},\n                filename=__file__\n            )\n            @triton.jit\n            ')
        compile_wrapper.splice(kernel.src, strip=True)
        from triton import JITFunction
        symbols_included = {original_name}

        def traverse(cur_kernel):
            for symbol_name in cur_kernel.fn.__code__.co_names:
                if symbol_name in symbols_included:
                    continue
                if symbol_name in cur_kernel.fn.__globals__:
                    symbol = cur_kernel.fn.__globals__[symbol_name]
                    if isinstance(symbol, JITFunction):
                        compile_wrapper.newline()
                        compile_wrapper.writeline('@triton.jit')
                        compile_wrapper.splice(symbol.src, strip=True)
                        symbols_included.add(symbol_name)
                        traverse(symbol)
                    elif isinstance(symbol, (int, str, bool)):
                        compile_wrapper.newline()
                        compile_wrapper.writeline(f'{symbol_name} = {symbol!r}')
                        symbols_included.add(symbol_name)
        traverse(kernel)
        compile_wrapper.writeline("''')")
        _, lineno = inspect.getsourcelines(kernel.fn)
        srcfile = inspect.getsourcefile(kernel.fn)
        metadata = f'# Original path: {srcfile}:{lineno}'
        self.define_kernel(name, compile_wrapper.getvalue(), metadata)
        return name

    def generate_numel_expr(self, kernel_name: str, tree):
        expr = f'{kernel_name}_{tree.prefix}numel'
        if expr not in self.kenel_numel_expr:
            self.kenel_numel_expr.add(expr)
            self.writeline(f'{self.declare}{expr} = {self.expr_printer(tree.numel)}{self.ending}')
        else:
            self.writeline(f'{expr} = {self.expr_printer(tree.numel)}{self.ending}')
        return SymbolicCallArg(expr, tree.numel)

    def wrap_kernel_call(self, name, call_args):
        return f'{name}({', '.join(call_args)}){self.ending}'

    def generate_profiler_mark_wrapper_call(self, stack):
        self.wrapper_call.writeline('from torch.profiler import record_function')
        self.wrapper_call.writeline(f"with record_function('graph_{V.graph.graph_id}_inductor_wrapper_call'):")
        stack.enter_context(self.wrapper_call.indent())

    def generate_start_graph(self):
        self.wrapper_call.writeline('start_graph()')

    def generate_end_graph(self):
        self.wrapper_call.writeline('end_graph()')

    def generate_default_grid(self, name: str, grid_args: List[Any]):
        return grid_args

    def generate_kernel_call(self, name, call_args, grid=None, device_index=None, cuda=True, triton=True):
        """
        Generates kernel call code.

        cuda: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if cuda:
            call_args_str = ', '.join((pexpr(item) for item in call_args))
            stream_name = self.write_get_raw_stream(V.graph.scheduler.current_device.index)
            if triton:
                grid_str = ', '.join((pexpr(item) for item in grid))
                self.writeline(f'{name}.run({call_args_str}, grid=grid({grid_str}), stream={stream_name})')
            else:
                stream_ptr = f'c_void_p({stream_name})'
                self.writeline(f'{name}.{name}({call_args_str}, {stream_ptr})')
        else:
            self.writeline(self.wrap_kernel_call(name, call_args))

    def writeline(self, line):
        self.lines.append(line)

    def enter_context(self, ctx):
        self.lines.append(LineContext(ctx))

    def val_to_cpp_arg_str(self, type_, val, is_legacy_abi) -> str:
        raise NotImplementedError()

    def val_to_arg_str(self, s):
        if isinstance(s, SymTypes):
            return pexpr(sympy.expand(repr(s)))
        elif isinstance(s, sympy.Expr):
            return pexpr(s)
        elif isinstance(s, (tuple, list)):

            @dataclasses.dataclass
            class Shim:
                ref: Any

                def __repr__(self):
                    return self.ref
            return repr(type(s)((Shim(self.val_to_arg_str(a)) for a in s)))
        elif isinstance(s, torch._ops.OpOverload):
            return _get_qualified_name(s)
        elif isinstance(s, (ComputedBuffer, InputBuffer, ReinterpretView)):
            return s.codegen_reference()
        else:
            return repr(s)

    def make_buffer_allocation(self, buffer):
        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        return self.make_allocation(buffer.get_name(), device, dtype, shape, stride)

    def make_allocation(self, name, device, dtype, shape, stride):
        try:
            expected = tuple(ir.make_contiguous_strides_for(shape))
        except Exception:
            expected = None
        if stride == expected:
            return f"{name} = empty({self.codegen_shape_tuple(shape)}, device='{device.type}', dtype={dtype})"
        else:
            return f"{name} = empty_strided({self.codegen_shape_tuple(shape)}, {self.codegen_shape_tuple(stride)}, device='{device.type}', dtype={dtype})"

    def make_tensor_alias(self, new_name, old_name, comment=''):
        return f'{self.declare}{new_name} = {old_name}{self.ending}  {self.comment} {comment}'

    def make_buffer_free(self, buffer):
        return f'del {buffer.get_name()}'

    def make_free_by_names(self, names_to_del: List[str]):
        return f'del {', '.join((name for name in names_to_del))}'

    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        return f'{self.declare}{new_name} = {old_name}{del_line}{self.ending}  {self.comment} reuse'

    def make_buffer_reuse(self, old, new, delete_old: bool):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ';'
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f'; {self.make_buffer_free(old)}'
        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            if old_name in self.cached_thread_locals:
                self.cached_thread_locals.add(new_name)
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)
        reinterpret_view = self.codegen_reinterpret_view(old, new.get_size(), new.get_stride(), 0, self.wrapper_call)
        if reinterpret_view in self.cached_thread_locals:
            self.cached_thread_locals.add(new_name)
        return f'{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse'

    def codegen_deferred_allocation(self, name, layout):
        self.writeline(DeferredLine(name, f'{self.declare}{name} = {layout.view.codegen_reference()}{self.ending}  {self.comment} alias'))

    def codegen_allocation(self, buffer):
        assert buffer.get_workspace_size() == 0, 'Only support zero workspace size for now!'
        name = buffer.get_name()
        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)
        if isinstance(buffer, (ir.ExternKernelAlloc, ir.MultiOutput)):
            return
        layout = buffer.get_layout()
        if isinstance(layout, ir.MutationLayout):
            return
        if isinstance(layout, ir.AliasedLayout):
            assert isinstance(layout.view, ir.ReinterpretView), f'unexpected {type(layout.view)}: {layout.view}'
            self.codegen_allocation(layout.view.data)
            self.codegen_deferred_allocation(name, layout)
            return
        self.writeline(AllocateLine(self, buffer))

    def codegen_free(self, buffer):
        assert buffer.get_workspace_size() == 0, 'Only support zero workspace size for now!'
        name = buffer.get_name()
        if isinstance(buffer, ir.InputBuffer):
            self.writeline(self.make_buffer_free(buffer))
            return
        if not self.can_reuse(buffer):
            return
        self.freed.add(name)
        self.writeline(FreeIfNotReusedLine(self, buffer))

    def can_reuse(self, input_buffer, output_buffer=None):
        name = input_buffer.get_name()
        if name in V.graph.removed_buffers or name in V.graph.graph_inputs or name in V.graph.constants or (name in V.graph.never_reuse_buffers) or (name in self.freed):
            return False
        return True

    def did_reuse(self, buffer, reused_buffer):
        return buffer.get_name() in self.reuses and self.reuses[buffer.get_name()] == reused_buffer.get_name()

    def codegen_inplace_reuse(self, input_buffer, output_buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.reuses[output_buffer.get_name()] = input_buffer.get_name()
        self.writeline(ReuseLine(self, input_buffer, output_buffer))

    def codegen_unbacked_symbol_decl(self, symbol):
        name = str(symbol)
        if name in self.unbacked_symbol_decls:
            return name
        else:
            self.unbacked_symbol_decls.add(name)
            return self.declare + name