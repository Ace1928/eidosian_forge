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
class InputsKernel(Buffer):
    inputs: List[Buffer]

    def get_read_writes_input(self, x):
        return dependencies.StarDep(x.get_name())

    def get_read_writes(self):
        star_dep = []
        for input in self.inputs:
            if isinstance(input, list):
                star_dep.extend([self.get_read_writes_input(x) for x in input])
            else:
                star_dep.append(self.get_read_writes_input(input))
        return dependencies.ReadWrites(set(star_dep), {dependencies.StarDep(self.get_name())}, set(), [], None, op_counts=collections.Counter())

    @staticmethod
    def unwrap_storage_for_input(x):
        if isinstance(x, TensorBox):
            x = x.data
        if isinstance(x, StorageBox):
            x = x.data
        if isinstance(x, BaseView) and (not isinstance(x, ReinterpretView)):
            x = ExternKernel.realize_input(x)
        assert isinstance(x, (Buffer, ReinterpretView)), x
        return x

    @staticmethod
    def unwrap_storage(inputs):
        inputs_new = []
        for x in inputs:
            if isinstance(x, list):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self):
        return True