from __future__ import annotations
import functools
import warnings
from collections.abc import Mapping, Callable
from copy import deepcopy
from typing import Any
import numpy as np
import symengine as sym
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
def _wrapped_lamb(*args):
    if isinstance(args[0], np.ndarray):
        t = args[0]
        args = np.hstack((t.reshape(t.size, 1), np.tile(args[1:], t.size).reshape(t.size, len(args) - 1)))
    return lamb(args)