import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DefMeasureCalibration(AbstractInstruction):

    def __init__(self, qubit: Union[Qubit, FormalArgument], memory_reference: Optional[MemoryReference], instrs: List[AbstractInstruction]):
        self.name = 'MEASURE'
        self.qubit = qubit
        self.memory_reference = memory_reference
        self.instrs = instrs

    def out(self) -> str:
        ret = f'DEFCAL MEASURE {self.qubit}'
        if self.memory_reference is not None:
            ret += f' {self.memory_reference}'
        ret += ':\n'
        for instr in self.instrs:
            ret += f'    {instr.out()}\n'
        return ret