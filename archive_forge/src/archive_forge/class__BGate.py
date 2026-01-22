from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
class _BGate(ops.Gate):
    """Single qubit gates and two of these can achieve any kak coefficients.

    References:
        Minimum construction of two-qubit quantum operations
        https://arxiv.org/abs/quant-ph/0312193
    """

    def num_qubits(self) -> int:
        return 2

    def _decompose_(self, qubits):
        a, b = qubits
        return [ops.XX(a, b) ** (-0.5), ops.YY(a, b) ** (-0.25)]