import copy
from typing import Union
from scipy.linalg import fractional_matrix_power
import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import (
from pennylane.ops.identity import Identity
from pennylane.queuing import QueuingManager, apply
from .symbolicop import ScalarSymbolicOp
class PowObs(Pow, Observable):
    """A child class of ``Pow`` that also inherits from ``Observable``."""

    def __new__(cls, *_, **__):
        return object.__new__(cls)