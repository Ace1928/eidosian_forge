from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class ApplyOp(cirq.Operation):

    def __init__(self, q):
        self.q = q

    @property
    def qubits(self):
        return (self.q,)

    def with_qubits(self, *new_qubits):
        return ApplyOp(*new_qubits)

    def _apply_unitary_(self, args):
        return cirq.apply_unitary(cirq.X(self.q), args)