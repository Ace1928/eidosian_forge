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
class InPlaceCollectiveKernel(CollectiveKernel):
    """
    InPlaceCollectiveKernel are those with in-out arguments such as all_reduce.
    Extend this kernel if your collective needs to modify its inputs in-place.
    """

    def __init__(self, layout, inputs, constant_args):
        super().__init__(layout, inputs, constant_args)

    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    def codegen_output(self, wrapper, output_name, input_names):
        if len(input_names) > 1:
            wrapper.writeline(f'{output_name} = [{','.join(input_names)}] ')
        else:
            wrapper.writeline(f'{output_name} = {input_names[0]}')