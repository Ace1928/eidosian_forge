import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class ClassicalMove(AbstractInstruction):
    """
    The MOVE instruction.

    WARNING: In pyQuil 2.0, the order of operands is as MOVE <target> <source>.
             In pyQuil 1.9, the order of operands was MOVE <source> <target>.
             These have reversed.
    """
    op = 'MOVE'

    def __init__(self, left: MemoryReference, right: Union[MemoryReference, int, float]):
        if not isinstance(left, MemoryReference):
            raise TypeError('Left operand of MOVE should be an MemoryReference.  Note that the order of the operands in pyQuil 2.0 has reversed from the order of pyQuil 1.9 .')
        if not isinstance(right, MemoryReference) and (not isinstance(right, int)) and (not isinstance(right, float)):
            raise TypeError('Right operand of MOVE should be an MemoryReference or a numeric literal')
        self.left = left
        self.right = right

    def out(self) -> str:
        return '%s %s %s' % (self.op, self.left, self.right)