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
class TemplateBuffer(Buffer):
    """
    Represents a Triton (in the future other type) of template operator
    that we can fuse an epilogue onto.
    """

    def __init__(self, layout, inputs, make_kernel_render):
        super().__init__(name=None, layout=layout)
        self.inputs = InputsKernel.unwrap_storage(inputs)
        self.make_kernel_render = make_kernel_render
        self.name = V.graph.register_buffer(self)

    def get_read_writes(self):
        return self.normalized_read_writes()

    def normalized_read_writes(self):
        name = self.get_name()
        indexer = self.layout.make_indexer()

        def dummy(index, rindex):
            assert len(rindex) == 0
            return ops.store(name, indexer(index), 'fake')
        deps = dependencies.extract_read_writes(dummy, self.get_size(), (), normalize=True)
        deps.reads = {dependencies.StarDep(x.get_name()) for x in self.inputs}
        return deps

    def get_reduction_size(self):
        return 1

    def get_reduction_type(self):
        return None

    def is_no_op(self):
        return False

    def should_allocate(self):
        return True

    def simplify_and_reorder(self):
        return ((self.get_size(), ()), None)