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
class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs, dim):
        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        assert 0 <= dim < len(new_size)
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            assert len(input_size) == len(new_size)
            assert inputs[i].get_dtype() == dtype
            assert inputs[i].get_device() == device
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.guard_equals(new_size[j], input_size[j])
            offsets_end.append(new_size[dim])
        output_stride = FlexibleLayout.contiguous_strides(new_size)
        for i in range(len(inputs)):
            x = inputs[i]
            if is_storage_and_layout(x):
                layout = x.get_layout()
                if isinstance(layout, FixedLayout) and layout.is_channels_last_contiguous():
                    output_stride = make_channels_last_strides_for(new_size)
                    break
        concat_kernel = ConcatKernel(name=None, layout=FixedLayout(device=device, dtype=dtype, size=new_size, stride=output_stride), inputs=[])
        kernel = StorageBox(concat_kernel)
        buffer_names = []
        for i in range(len(inputs)):
            input_buffer = cls.realize_into(inputs[i], SliceView.create(kernel, dim, offsets_start[i], offsets_end[i]))
            concat_kernel.inputs.append(input_buffer)
            if isinstance(inputs[i].data, BaseView):
                input_unwrapped = inputs[i].data.unwrap_view()
            else:
                input_unwrapped = inputs[i].data
            if input_unwrapped.is_input_buffer() and inputs[i].get_device().type == 'cuda' and (not is_dynamic(input_buffer)):
                buffer_names.append(input_buffer.get_name())
        if len(buffer_names) > 1:
            V.graph.register_list(buffer_names)
        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
        return kernel

    @classmethod
    def can_realize_into_without_copy(cls, src):
        if isinstance(src, TensorBox):
            return cls.can_realize_into_without_copy(src.data)
        return isinstance(src.data.layout, FlexibleLayout) and (not isinstance(src.data, ExternKernelAlloc))

    @classmethod
    def realize_into(cls, src, dst):
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(storage, layout)
        assert isinstance(dst, ReinterpretView), dst
        if isinstance(src, TensorBox):
            return cls.realize_into(src.data, dst)
        if isinstance(src, StorageBox):
            src.realize()
            assert hasattr(src.data, 'layout')
            if cls.can_realize_into_without_copy(src):
                src.data.layout = AliasedLayout(dst)
                return src.data
        pw = Pointwise.create(device=src.get_device(), dtype=src.get_dtype(), inner_fn=src.make_loader(), ranges=[V.graph.sizevars.guard_equals(a, b) for a, b in zip(src.get_size(), dst.get_size())])
        return cls.realize_into(pw, dst)

    def should_allocate(self):
        return True