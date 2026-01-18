import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('psum, expected_qubits', ((cirq.Z(q1), (q1,)), (cirq.X(q0) + cirq.Y(q0), (q0,)), (cirq.X(q0) + cirq.Y(q2), (q0, q2)), (cirq.X(q2) + cirq.Y(q0), (q0, q2)), (cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3), (q0, q1, q3))))
def test_pauli_sum_qubits(psum, expected_qubits):
    assert psum.qubits == expected_qubits