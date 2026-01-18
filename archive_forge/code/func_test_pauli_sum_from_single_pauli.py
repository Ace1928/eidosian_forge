import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_pauli_sum_from_single_pauli():
    q = cirq.LineQubit.range(2)
    psum1 = cirq.X(q[0]) + cirq.Y(q[1])
    assert psum1 == cirq.PauliSum.from_pauli_strings([cirq.X(q[0]) * 1, cirq.Y(q[1]) * 1])
    psum2 = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Y(q[1])
    assert psum2 == cirq.PauliSum.from_pauli_strings([cirq.X(q[0]) * cirq.X(q[1]), cirq.Y(q[1]) * 1])
    psum3 = cirq.Y(q[1]) + cirq.X(q[0]) * cirq.X(q[1])
    assert psum3 == psum2