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
class CollectiveKernel(ExternKernel):
    """
    Each collective should follow the pattern:
    - extend InPlaceCollectiveKernel or OutOfPlaceCollectiveKernel.
    - the kernel delegates into c10d processgroup, which returns a 'work' obj
    - the work obj is registered via _register_tensor_work so it can be waited on later
    """

    def __init__(self, layout, inputs, constant_args):
        super().__init__(None, layout, inputs, constant_args)
        self.name = V.graph.register_buffer(self)

    def should_emit_register_tensor_work(self):
        return True

    def should_emit_find_or_create_pg(self):
        return True

    def codegen_collective(self, wrapper, output_name, input_names):
        raise NotImplementedError('Must implement')

    def codegen_output(self, wrapper, output_name, input_names):
        raise NotImplementedError('Must implement')

    @classmethod
    def wrap_inputs_as_inplace(cls, inputs):

        def wrap_input(var):
            op = InPlaceHint(FlexibleLayout(var.get_device(), var.get_dtype(), var.get_size()), var)
            return TensorBox.create(op)
        return list(map(wrap_input, inputs))

    def codegen(self, wrapper):
        wrapper.add_import_once('import torch.distributed as dist')
        wrapper.add_import_once('import torch.distributed.distributed_c10d as c10d')
        wrapper.add_import_once('import torch.distributed._functional_collectives_impl as fun_col_impl')
        input_names = [t.codegen_reference() for t in self.inputs]
        output_name = self.get_name()
        tag, ranks, group_size = self.constant_args
        if self.should_emit_find_or_create_pg():
            wrapper.writeline(f"{output_name}_pg = c10d._find_or_create_pg_by_ranks_and_tag('{tag}', {ranks}, {group_size})")
        self.codegen_output(wrapper, output_name, input_names)
        self.codegen_collective(wrapper, output_name, input_names)
        if self.should_emit_register_tensor_work():
            wrapper.writeline(f'fun_col_impl._register_tensor_work({output_name}, {output_name}_work)')