import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
@versionadded(version='3.5.1', reason='The correct instruction is SWAP-PHASES, not SWAP-PHASE')
class SwapPhases(AbstractInstruction):

    def __init__(self, frameA: Frame, frameB: Frame):
        self.frameA = frameA
        self.frameB = frameB

    def out(self) -> str:
        return f'SWAP-PHASES {self.frameA} {self.frameB}'

    def get_qubits(self, indices: bool=True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frameA, indices) | _get_frame_qubits(self.frameB, indices)