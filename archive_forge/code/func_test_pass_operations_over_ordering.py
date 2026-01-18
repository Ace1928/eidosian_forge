import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pass_operations_over_ordering():

    class OrderSensitiveGate(cirq.Gate):

        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** (-0.5), cirq.CNOT(*qubits)]
    a, b = cirq.LineQubit.range(2)
    inp = cirq.Z(b)
    out1 = inp.pass_operations_over([OrderSensitiveGate().on(a, b)])
    out2 = inp.pass_operations_over([cirq.CNOT(a, b), cirq.Y(a) ** (-0.5)])
    out3 = inp.pass_operations_over([cirq.CNOT(a, b)]).pass_operations_over([cirq.Y(a) ** (-0.5)])
    assert out1 == out2 == out3 == cirq.X(a) * cirq.Z(b)