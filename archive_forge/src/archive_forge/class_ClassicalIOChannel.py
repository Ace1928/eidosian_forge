from __future__ import annotations
from abc import ABCMeta
from typing import Any
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
class ClassicalIOChannel(Channel, metaclass=ABCMeta):
    """Base class of classical IO channels. These cannot have instructions scheduled on them."""
    pass