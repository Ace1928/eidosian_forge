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
class DeviceCopy(ExternKernelOut):

    @classmethod
    def create(cls, x, device):
        if not x.is_extern() and all((r.name in V.graph.constants and isinstance(r, dependencies.MemoryDep) for r in x.get_reads())):
            return x.constant_to_device(device)
        V.graph.add_device_info(device)
        V.graph.add_device_info(x.get_device())
        developer_warning('DeviceCopy in input program')
        return DeviceCopy(FlexibleLayout(device=device, dtype=x.get_dtype(), size=x.get_size()), [cls.realize_input(x)])

    def codegen(self, wrapper):
        args = self.codegen_args()
        assert len(args) == 1
        if self.output_view:
            wrapper.codegen_device_copy(args[0], self.output_view.codegen_reference())
        else:
            wrapper.codegen_device_copy(args[0], self.codegen_reference())