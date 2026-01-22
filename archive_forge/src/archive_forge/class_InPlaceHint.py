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
class InPlaceHint(ExternKernel):
    """
    Helper OP to encode an in/out argument that tries to make it inplace whenever possible.
    Wrap the input of your inplace op to enable this behavior.

    The design is based on two key decisions:
    - this node is responsible for allocating the in/out buffer used by the collective.
        This is controlled by the ``should_allocate`` method that returns True here and
        False for the collective node
    - The scheduler special-case this node and enable it to reuse its input.
    """

    def codegen(self, wrapper):
        input_name = self.inputs[0].codegen_reference()
        output_name = self.get_name()
        if not wrapper.did_reuse(self, self.inputs[0]):
            wrapper.writeline(f'{output_name}.copy_({input_name}) #no reuse')

    def __init__(self, layout, input):
        input = self.realize_input(input)
        super().__init__(None, layout, self.unwrap_storage([input]), ())
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return True