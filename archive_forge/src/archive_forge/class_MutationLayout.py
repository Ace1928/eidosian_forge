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
class MutationLayout(Layout):

    def __init__(self, target: IRNode):
        super().__init__(target.get_device(), target.get_dtype(), target.get_size(), None)
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @Layout.stride.getter
    def stride(self):
        return self.real_layout().stride

    def storage_size(self) -> sympy.Expr:
        return self.real_layout().storage_size()

    def get_buffer(self) -> 'Buffer':

        def unwrap_views(target):
            if isinstance(target, MutationLayout):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target
        result = unwrap_views(self.target)
        assert isinstance(result, Buffer), 'MutationLayout must refer to a buffer'
        return result

    def real_layout(self):
        return self.get_buffer().layout

    @classmethod
    def realize_into(cls, src, dst, unsafe_alias=False):
        dst.realize()
        V.graph.mark_buffer_mutated(dst.get_name())
        if isinstance(src, TensorBox):
            src = src.data
        src.realize_hint()
        if not unsafe_alias:
            src = Pointwise.create(device=src.get_device(), dtype=src.get_dtype(), inner_fn=src.make_loader(), ranges=[V.graph.sizevars.guard_equals(a, b) for a, b in zip(src.get_size(), dst.get_size())]).data
        src.realize()
        assert isinstance(src.data.layout, FlexibleLayout)
        src.data.layout = MutationLayout(dst)
        return src.data

    def as_fixed(self):
        return self

    def make_indexer(self):
        return self.target.make_indexer()