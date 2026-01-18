import itertools
import math
import numpy as np
import pytest
import sympy
import cirq
import cirq.contrib.quimb as ccq
import cirq.testing
from cirq import value
def test_various_gates_1d_flip():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q1), cirq.CNOT(q1, q0))
    assert_same_output_as_dense(circuit=circuit, qubit_order=[q0, q1])
    assert_same_output_as_dense(circuit=circuit, qubit_order=[q1, q0])