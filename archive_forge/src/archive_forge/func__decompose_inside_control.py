from typing import (
import numpy as np
import sympy
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
def _decompose_inside_control(self, target1: 'cirq.Qid', control: 'cirq.Qid', target2: 'cirq.Qid') -> 'cirq.OP_TREE':
    """A decomposition assuming the control separates the targets.

        target1: ─@─X───────T──────@────────@─────────X───@─────X^-0.5─
                  │ │              │        │         │   │
        control: ─X─@─X─────@─T^-1─X─@─T────X─@─X^0.5─@─@─X─@──────────
                      │     │        │        │         │   │
        target2: ─────@─H─T─X─T──────X─T^-1───X─T^-1────X───X─H─S^-1───
        """
    a, b, c = (target1, control, target2)
    yield common_gates.CNOT(a, b)
    yield common_gates.CNOT(b, a)
    yield common_gates.CNOT(c, b)
    yield common_gates.H(c)
    yield common_gates.T(c)
    yield common_gates.CNOT(b, c)
    yield common_gates.T(a)
    yield (common_gates.T(b) ** (-1))
    yield common_gates.T(c)
    yield common_gates.CNOT(a, b)
    yield common_gates.CNOT(b, c)
    yield common_gates.T(b)
    yield (common_gates.T(c) ** (-1))
    yield common_gates.CNOT(a, b)
    yield common_gates.CNOT(b, c)
    yield (pauli_gates.X(b) ** 0.5)
    yield (common_gates.T(c) ** (-1))
    yield common_gates.CNOT(b, a)
    yield common_gates.CNOT(b, c)
    yield common_gates.CNOT(a, b)
    yield common_gates.CNOT(b, c)
    yield common_gates.H(c)
    yield (common_gates.S(c) ** (-1))
    yield (pauli_gates.X(a) ** (-0.5))