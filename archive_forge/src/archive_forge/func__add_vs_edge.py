import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
def _add_vs_edge(self, builder, src, dst, **attrs):
    is_effect = isinstance(src, ValueState) and src.is_effect or (isinstance(dst, ValueState) and dst.is_effect)
    if isinstance(src, ValueState):
        src = src.short_identity()
    if isinstance(dst, ValueState):
        dst = dst.short_identity()
    kwargs = attrs
    if is_effect:
        kwargs['kind'] = 'effect'
    builder.graph.add_edge(src, dst, **kwargs)