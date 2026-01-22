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
class AccumulateGrad(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """
    kernel = 'inductor_ops.accumulate_grad_'

    def codegen(self, wrapper):
        variable, new_grad = (t.codegen_reference() for t in self.inputs)
        wrapper.writeline(f'{self.kernel}({variable}, {new_grad})')

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        return {}

    def __init__(self, variable, new_grad):
        super().__init__(None, NoneLayout(variable.get_device()), self.unwrap_storage([variable, new_grad]))
        self.name = V.graph.register_buffer(self)
        mark_node_as_mutating(self, variable)