import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
@dataclasses.dataclass
class ExternKernel(InputsKernel):
    constant_args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    output_view: Optional[ReinterpretView] = None
    ordered_kwargs_for_cpp_kernel: Iterable[str] = dataclasses.field(default_factory=list)

    def decide_layout(self):
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(self, wrapper):
        origin_str, detailed_origin_str = get_kernel_metadata(self, wrapper)
        if origin_str:
            wrapper.writeline(origin_str)

    def codegen(self, wrapper):
        raise NotImplementedError()

    @staticmethod
    def copy_input(x):
        pw = Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=x.make_loader(), ranges=x.get_size(), origin_node=x.get_origin_node(), traceback=x.get_traceback())
        pw.realize()
        return pw

    @classmethod
    def process_kernel(cls, kernel, *args, **kwargs):
        binded_args = signature(kernel).bind(*args, **kwargs).arguments
        args_flat, args_spec = pytree.tree_flatten(binded_args)
        is_arg_tensor = []
        tensor_args = []
        non_tensor_args: List[Any] = []
        for arg in args_flat:
            is_arg_tensor.append(isinstance(arg, IRNode))
            if is_arg_tensor[-1]:
                tensor_args.append(arg)
            else:
                if isinstance(arg, sympy.Expr):
                    arg = V.graph.sizevars.shape_env.create_symintnode(arg, hint=None)
                non_tensor_args.append(arg)

        def unflatten_args(new_tensor_args, new_non_tensor_args):
            result = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in is_arg_tensor:
                if is_tensor:
                    result.append(next(it_tensors))
                else:
                    result.append(next(it_non_tensors))
            r = pytree.tree_unflatten(result, args_spec)
            return (r.get('args', []), r.get('kwargs', {}))
        tensor_args = [cls.realize_input(x) for x in tensor_args]
        for x in tensor_args:
            if is_storage_and_layout(x):
                as_storage_and_layout(x, freeze=True)
        example_args = []
        for x in tensor_args:
            if x.get_name() in V.graph.constants:
                example_args.append(V.graph.constants[x.get_name()])
            else:
                example_args.append(ir_node_to_tensor(x, guard_shape=True))
        new_args, new_kwargs = unflatten_args(example_args, non_tensor_args)
        example_output = kernel(*new_args, **new_kwargs)
        example_out_li = [example_output] if not isinstance(example_output, (list, tuple)) else example_output
        for t in example_out_li:
            if isinstance(t, torch.Tensor) and t.is_sparse:
                V.graph.disable_cudagraphs = True
                msg = 'sparsity not handled. Please file issue for sparse inference weights.'
                if (stack_trace := V.graph.current_node.meta.get('stack_trace', None)):
                    msg = f'{msg} Found from : \n {stack_trace}'
                V.graph.disable_cudagraphs_reason = msg
        if maybe_free_unbacked_symbols(example_output):
            example_output = V.graph.current_node.meta['val']
        return (example_output, tensor_args, non_tensor_args, unflatten_args)

    @classmethod
    def convert_to_reinterpret_view(cls, x):
        """
        In order to pass this to an extern kernel we need a
        ReinterpretView not a View.  This allows us to avoid some
        unneeded copies.
        """
        assert isinstance(x, BaseView)
        if isinstance(x, ReinterpretView):
            return x
        x.unwrap_view().freeze_layout()
        index_args, var_ranges = dependencies.index_vars_squeeze(x.get_size(), prefix='r')
        range_vars = index_args[0]
        index = x.make_indexer()(range_vars)
        index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        strides = V.graph.sizevars.stride_vars(index, range_vars)
        offset = V.graph.sizevars.offset_var(index, range_vars)
        expected = sympy_dot(range_vars, strides) + offset
        if index != expected:
            log.debug('convert_to_reinterpret_view failed: stride=%s offset=%s index=%s', strides, offset, index)
            raise NotImplementedError()
        return ReinterpretView(data=x.data, layout=FixedLayout(device=x.get_device(), dtype=x.get_dtype(), size=x.get_size(), stride=strides, offset=offset))

    @classmethod
    def realize_input(cls, x):
        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, (sympy.Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(x)
        if isinstance(x, Constant):
            return V.graph.add_tensor_constant(torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device()))
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return x
        if isinstance(x, BaseView):
            x.realize()
            if is_storage_and_layout(x.unwrap_view()):
                try:
                    return cls.convert_to_reinterpret_view(x)
                except NotImplementedError:
                    pass
        if isinstance(x, StorageBox):
            x.realize()
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x):
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        return cls.copy_input(x)

    @classmethod
    def require_stride_order(cls, x, order):
        if x.get_numel() == 0:
            return x
        if is_storage_and_layout(x):
            while isinstance(x.get_layout(), AliasedLayout):
                x = x.get_layout().view
            if isinstance(x.get_layout(), FlexibleLayout):
                as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=order)
                return x
            elif isinstance(x.get_layout(), FixedLayout) and x.get_layout().is_stride_ordered(order):
                return x
            elif isinstance(x.get_layout(), MutationLayout):
                if isinstance(x.get_layout().real_layout(), FlexibleLayout):
                    raise AssertionError("the MutationLayout's real layout shouldn't be FlexibleLayout")
                elif isinstance(x.get_layout().real_layout(), FixedLayout) and x.get_layout().real_layout().is_stride_ordered(order):
                    return x
        if isinstance(x, InputBuffer) and x.get_layout().is_stride_ordered(order):
            return x
        if isinstance(x, TensorBox) and isinstance(x.data, BaseView) and (not isinstance(x.data, ReinterpretView)) and is_storage_and_layout(x.unwrap_view()) and (not isinstance(x.unwrap_view().data, ExternKernelAlloc)):
            try:
                x.data = cls.convert_to_reinterpret_view(x.data)
                return cls.require_stride_order(x, order)
            except NotImplementedError:
                pass
        x = cls.copy_input(x)
        as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=order)
        assert is_stride_order_storage_and_layout(x, order)
        return x

    @classmethod
    def require_channels_last(cls, x):
        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_contiguous(cls, x):
        return cls.require_stride_order(x, list(reversed(range(len(x.get_size())))))

    def apply_constraint(self):
        pass

    def codegen_const_args(self):
        return map(V.graph.wrapper_code.val_to_arg_str, self.constant_args)

    def codegen_args(self):
        args = []
        for x in self.inputs:
            if isinstance(x, list):
                names = [i.codegen_reference() for i in x]
                codegen_reference = f'[{', '.join(names)}]'
                args.append(codegen_reference)
            else:
                args.append(x.codegen_reference())
        args.extend(self.codegen_const_args())
        return args

    def get_kwargs_value(self, arg_name):
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        if hasattr(self, 'kwargs_default_value') and arg_name in self.kwargs_default_value:
            return self.kwargs_default_value.get(arg_name).get('value')
        raise AssertionError(f'arg {arg_name} not found in self.kwargs or self.kwargs_default_value')

    def is_legacy_abi_kernel(self):
        return False

    def codegen_kwargs(self):
        if V.graph.cpp_wrapper:
            if self.kwargs and (not self.ordered_kwargs_for_cpp_kernel):
                raise AssertionError('ordered_kwargs_for_cpp_kernel is missing')
            kwargs = []
            for arg_name in self.ordered_kwargs_for_cpp_kernel:
                v = self.get_kwargs_value(arg_name)
                if isinstance(v, sympy.Expr):
                    kwargs.append(v)
                else:
                    if hasattr(self, 'kwargs_default_value'):
                        type_ = self.kwargs_default_value.get(arg_name).get('type')
                    else:
                        type_ = None
                    kwargs.append(V.graph.wrapper_code.val_to_cpp_arg_str(type_, v, self.is_legacy_abi_kernel()))
        else:
            kwargs = [f'{k}={V.graph.wrapper_code.val_to_arg_str(v)}' for k, v in self.kwargs.items()]
        return kwargs

    def codegen_size_asserts(self, wrapper):
        if config.size_asserts and (not V.graph.cpp_wrapper):
            size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
            stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
            wrapper.writeline(f'assert_size_stride({self.get_name()}, {size}, {stride})')

    def get_group_stride(self):
        """
        get output sizes and strides, for template_codegen
        """
        _size = self.get_size()
        _stride = self.get_stride()
        return ([_size, []], _stride)

    def canonicalize(self):
        """
        Manually get canonicalization of the output index
        """
        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        strides = [sizevars.size_hint(x) for x in strides]
        index_vars = [sympy_symbol(f'd{i}') for i in range(len(sizes))]
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for idx, pos in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(index_vars, sizes, [index])
        _, add_var = var_builder('c')
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))
        index = sympy_subs(sympy.expand(index), replacement)
        return (index, tuple(new_sizes))

    def get_unbacked_symbol_uses(self):
        r = set()
        for arg in self.constant_args:
            r |= maybe_free_unbacked_symbols(arg)
        for arg in self.kwargs.values():
            r |= maybe_free_unbacked_symbols(arg)
        return r

    def __str__(self):
        kernel_name = getattr(self, 'kernel', None)
        lines = [f'kernel={kernel_name!r}']
        lines += [f'{field.name}={getattr(self, field.name)}' for field in dataclasses.fields(self)]
        lines.append(f'origin_node={self.origin_node!r}')
        return self.str_helper(lines)
    __repr__ = __str__