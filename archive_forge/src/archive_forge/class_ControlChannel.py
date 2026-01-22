from __future__ import annotations
from abc import ABCMeta
from typing import Any
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
class ControlChannel(PulseChannel):
    """Control channels provide supplementary control over the qubit to the drive channel.
    These are often associated with multi-qubit gate operations. They may not map trivially
    to a particular qubit index.
    """
    prefix = 'u'