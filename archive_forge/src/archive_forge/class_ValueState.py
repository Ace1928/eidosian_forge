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
@dataclass(frozen=True)
class ValueState:
    """For representing a RVSDG Value (State).

    For most compiler passes, Value and State can be treated as the same.
    """
    parent: Optional['Op']
    'Optional. The parent Op that output this ValueState.\n    '
    name: str
    'Name of the Value(State).\n    '
    out_index: int
    'The output port index in the parent Op.\n    '
    is_effect: bool = False
    'True if-and-only-if this is a state.\n    '

    def short_identity(self) -> str:
        args = f'{id(self.parent):x}, {self.name}, {self.out_index}'
        return f'ValueState({args})'

    def __hash__(self):
        return id(self)