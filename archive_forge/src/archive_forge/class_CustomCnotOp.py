import numpy as np
import pytest
import sympy
import cirq
class CustomCnotOp(cirq.Operation):

    def __init__(self, *qs: cirq.Qid):
        self.qs = qs

    def _unitary_(self):
        return cirq.unitary(cirq.CNOT)

    @property
    def qubits(self):
        return self.qs

    def with_qubits(self, *new_qubits):
        raise NotImplementedError()