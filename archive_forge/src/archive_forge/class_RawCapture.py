import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class RawCapture(AbstractInstruction):

    def __init__(self, frame: Frame, duration: float, memory_region: MemoryReference, nonblocking: bool=False):
        self.frame = frame
        self.duration = duration
        self.memory_region = memory_region
        self.nonblocking = nonblocking

    def out(self) -> str:
        result = 'NONBLOCKING ' if self.nonblocking else ''
        result += f'RAW-CAPTURE {self.frame} {self.duration} {self.memory_region.out()}'
        return result

    def get_qubits(self, indices: bool=True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)