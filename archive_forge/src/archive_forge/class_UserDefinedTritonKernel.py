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
class UserDefinedTritonKernel(ExternKernel):

    def get_kernel_and_configs(self):
        from triton.runtime.autotuner import Autotuner
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        kernel = kernel_side_table.get_kernel(self.kernel_idx)
        configs = []
        if isinstance(kernel, Autotuner):
            configs = kernel.configs
            kernel = kernel.fn
        return (kernel, configs)

    def codegen(self, wrapper):
        kernel, configs = self.get_kernel_and_configs()
        new_name = wrapper.define_user_defined_triton_kernel(kernel, configs, self.kwargs)
        args = self.codegen_kwargs()
        if V.graph.cpp_wrapper:
            args = [arg for i, arg in enumerate(args) if i not in kernel.constexprs]
        self.codegen_comment(wrapper)
        wrapper.generate_user_defined_triton_kernel(new_name, self.grid, configs, args)

    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    def get_unbacked_symbol_defs(self):
        return {}

    def get_mutation_names(self):
        return []

    def __init__(self, *, kernel_idx, grid, kernel_args):
        inputs = []
        kwargs = dict()
        constant_args = []
        for k, v in kernel_args.items():
            if isinstance(v, TensorBox):
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                inputs.append(t)
                kwargs[k] = t
            else:
                constant_args.append(v)
                kwargs[k] = v
        assert len(inputs) != 0
        device = inputs[0].get_device()
        super().__init__(None, NoneLayout(device), inputs, tuple(constant_args), kwargs)
        self.name = V.graph.register_buffer(self)
        self.kernel_idx = kernel_idx
        self.grid = grid
        kernel, _ = self.get_kernel_and_configs()
        self.ordered_kwargs_for_cpp_kernel = [arg for arg in kernel.arg_names if arg in kernel_args]
        mark_node_as_mutating(self, *[a for a in kernel_args.values() if isinstance(a, TensorBox)])

    def get_alias_names(self):
        return [i.get_name() for i in self.inputs]