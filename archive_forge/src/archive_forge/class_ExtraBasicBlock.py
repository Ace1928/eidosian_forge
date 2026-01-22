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
class ExtraBasicBlock(BasicBlock):
    inst_list: tuple[str, ...] = ()

    @classmethod
    def make(cls, label, jump_target, instlist):
        return ExtraBasicBlock(label, (jump_target,), inst_list=instlist)

    def __str__(self):
        args = '\n'.join((f'{inst})' for inst in self.inst_list))
        return f'ExtraBasicBlock({args})'