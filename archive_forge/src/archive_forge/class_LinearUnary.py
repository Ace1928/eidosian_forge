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
class LinearUnary(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        super().__init__(layout, inputs, constant_args, None, kernel='torch.ops.mkldnn._linear_pointwise', cpp_kernel='mkldnn::_linear_pointwise')
        self.cpp_kernel_key = 'linear_pointwise'
        self.cpp_op_schema = '\n            at::Tensor(\n                const at::Tensor& input_t,\n                const at::Tensor& weight_t,\n                const c10::optional<at::Tensor>& bias_opt,\n                c10::string_view attr,\n                torch::List<c10::optional<at::Scalar>> scalars,\n                c10::optional<c10::string_view> algorithm)'

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(self.get_name(), self.codegen_kernel_name(), self.codegen_args(), self.cpp_op_schema, self.cpp_kernel_key)

    @classmethod
    def create(cls, x, w, b, attr, scalars, algorithm):
        x = cls.require_contiguous(cls.realize_input(x))
        w = cls.require_contiguous(cls.realize_input(w))
        *m, ic = x.get_size()
        oc, ic = w.get_size()
        inputs = [x, w]
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        if b is not None:
            b = cls.require_contiguous(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, None)
        return LinearUnary(layout=FlexibleLayout(device=x.get_device(), dtype=x.get_dtype(), size=list(m) + [oc]), inputs=inputs, constant_args=constant_args)

    def apply_constraint(self):
        pass