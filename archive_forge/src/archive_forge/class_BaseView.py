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
class BaseView(IRNode):
    data: IRNode

    def make_reindexer(self):
        raise NotImplementedError(f'make_reindexer NYI on {self}')

    def make_indexer(self):
        inner = self.data.make_indexer()
        reindex = self.make_reindexer()

        def indexer(idx):
            return inner(reindex(idx))
        return indexer

    def make_loader(self):
        inner = self.data.make_loader()
        reindex = self.make_reindexer()

        def loader(idx):
            return inner(reindex(idx))
        return loader

    def get_dtype(self):
        return self.data.get_dtype()

    def get_layout(self):
        return self.data.get_layout()

    def get_device(self):
        return self.data.get_device()

    def get_origin_node(self):
        return None

    def get_name(self):
        return self.data.get_name()

    def mark_reuse(self, users):
        return self.data.mark_reuse(users)

    def has_exceeded_max_reads(self):
        return self.data.has_exceeded_max_reads()

    def realize(self):
        return self.data.realize()

    def realize_hint(self):
        return self.data.realize_hint()

    def get_storage_numel(self):
        return self.data.get_storage_numel()

    def is_extern(self):
        return self.data.is_extern()

    def get_reads(self):
        with patch.object(FlexibleLayout, 'allow_indexing', True):
            return extract_read_writes(self.make_loader(), self.get_size()).reads

    def unwrap_view(self):
        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, 'override_device', device)(loader)
        return Pointwise(device, self.get_dtype(), loader, self.get_size())