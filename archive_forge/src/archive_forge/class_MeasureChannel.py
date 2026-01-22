from __future__ import annotations
from abc import ABCMeta
from typing import Any
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
class MeasureChannel(PulseChannel):
    """Measure channels transmit measurement stimulus pulses for readout."""
    prefix = 'm'