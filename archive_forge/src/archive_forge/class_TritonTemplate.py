import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import patch
import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state
from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import ChoiceCaller, IndentedBuffer, KernelTemplate
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .utils import do_bench, Placeholder, sympy_dot, sympy_product, unique
from .virtualized import V
from . import lowering  # noqa: F401
class TritonTemplate(KernelTemplate):
    index_counter = itertools.count()
    all_templates: Dict[str, 'TritonTemplate'] = dict()

    def __init__(self, name: str, grid: Any, source: str, debug=False):
        super().__init__(name)
        self.grid = grid
        self.template = self._template_from_string(source)
        assert name not in self.all_templates, 'duplicate template name'
        self.all_templates[name] = self
        self.debug = debug

    def generate(self, input_nodes, layout, num_stages, num_warps, prefix_args=0, suffix_args=0, epilogue_fn=identity, **kwargs):
        assert self.template, 'requires jinja2'
        defines = StringIO()
        for name, val in kwargs.items():
            defines.write(f'    {name} : tl.constexpr = {val}\n')
        defines = defines.getvalue()
        fake_out = ir.Buffer('buf_out', layout)
        kernel_name = f'triton_{self.name}'
        numel = sympy_product(layout.size)
        buffers = itertools.chain(input_nodes, (fake_out,))
        if not TritonScheduling.can_use_32bit_indexing(numel, buffers):
            raise NotImplementedError('64-bit indexing is not yet implemented for triton templates')
        kernel_options = dict(input_nodes=input_nodes, defines=defines, num_stages=num_stages, num_warps=num_warps, grid_fn=self.grid, meta=kwargs, call_sizes=layout.size, prefix_args=prefix_args, suffix_args=suffix_args, epilogue_fn=epilogue_fn, index_dtype='tl.int32')
        with patch.object(V.graph, 'get_dtype', self._fake_get_dtype(fake_out)), TritonTemplateKernel(kernel_name=kernel_name, output_node=fake_out, use_jit=True, **kernel_options) as kernel:
            try:
                code = kernel.render(self.template, kwargs).finalize()
            except ZeroDivisionError:
                return None
            if self.debug:
                print('Generated Code:\n', code)
            extra = '-'.join([*[f'{kwarg}={repr(kwargs[kwarg])}' for kwarg in sorted(kwargs.keys())], f'num_stages={num_stages}', f'num_warps={num_warps}']) + '-'
            mod = PyCodeCache.load(code, extra)
            _, call_args, _ = kernel.args.python_argdefs()
        expected_args = list(unique((x.get_name() for x in input_nodes)))
        expected_args.extend([fake_out.get_name()])
        assert list(call_args)[:len(expected_args)] == expected_args, (call_args, expected_args)
        extra_args = V.graph.sizevars.size_hints(map(sympy.expand, call_args[len(expected_args):]), fallback=config.unbacked_symint_fallback)
        kernel_hash_name = f'triton_{self.name}_{next(self.index_counter)}'

        def make_kernel_render(out_node):
            kernel = TritonTemplateKernel(kernel_name=str(Placeholder.KERNEL_NAME), output_node=out_node, use_jit=False, **kernel_options)
            render = functools.partial(kernel.render, self.template, kwargs)
            return (kernel, render)
        assert mod.__file__ is not None
        grid = self.grid(*V.graph.sizevars.size_hints(layout.size, fallback=config.unbacked_symint_fallback), kwargs)
        bmreq = TritonBenchmarkRequest(module_path=mod.__file__, module_cache_key=mod.key, kernel_name=kernel_name, grid=grid, extra_args=extra_args, num_stages=num_stages, num_warps=num_warps, input_tensor_meta=TensorMeta.from_irnodes(input_nodes), output_tensor_meta=TensorMeta.from_irnodes(layout))
        return TritonTemplateCaller(kernel_hash_name, input_nodes, layout, make_kernel_render, extra.strip('-').replace('-', ', '), bmreq)