from __future__ import annotations
import warnings
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from enum import Enum
import numpy as np
from qiskit import circuit
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawings, types
from qiskit.visualization.timeline.stylesheet import QiskitTimelineStyle
def _check_bit_visible(self, bit: types.Bits) -> bool:
    """A helper function to check if the bit is visible.

        Args:
            bit: Bit object to test.

        Returns:
            Return `True` if the bit is visible.
        """
    _gates = [str(types.BoxType.SCHED_GATE.value), str(types.SymbolType.FRAME.value)]
    if bit in self.disable_bits:
        return False
    if self.formatter['control.show_idle']:
        return True
    for data in self._collections.values():
        if bit in data.bits and data.data_type in _gates:
            return True
    return False