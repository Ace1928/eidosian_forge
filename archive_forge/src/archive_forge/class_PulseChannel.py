from __future__ import annotations
from abc import ABCMeta
from typing import Any
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
class PulseChannel(Channel, metaclass=ABCMeta):
    """Base class of transmit Channels. Pulses can be played on these channels."""
    pass