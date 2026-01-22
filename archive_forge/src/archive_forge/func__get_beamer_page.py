import io
import itertools
import math
import re
from warnings import warn
import numpy as np
from qiskit.circuit import Clbit, Qubit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library.standard_gates import SwapGate, XGate, ZGate, RZZGate, U1Gate, PhaseGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.tools.pi_check import pi_check
from .qcstyle import load_style
from ._utils import (
def _get_beamer_page(self):
    """Get height, width & scale attributes for the beamer page."""
    pil_limit = 40000
    beamer_limit = 550
    aspect_ratio = self._sum_wire_heights / self._sum_column_widths
    margin_factor = 1.5
    height = min(self._sum_wire_heights * margin_factor, beamer_limit)
    width = min(self._sum_column_widths * margin_factor, beamer_limit)
    if height * width > pil_limit:
        height = min(np.sqrt(pil_limit * aspect_ratio), beamer_limit)
        width = min(np.sqrt(pil_limit / aspect_ratio), beamer_limit)
    height = max(height, 10)
    width = max(width, 10)
    return (height, width, self._scale)