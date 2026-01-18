import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_pauli_sum_pow():
    identity = cirq.PauliSum.from_pauli_strings([cirq.PauliString(coefficient=1)])
    psum1 = cirq.X(q0) + cirq.Y(q0)
    assert psum1 ** 2 == psum1 * psum1
    assert psum1 ** 2 == 2 * identity
    psum2 = cirq.X(q0) + cirq.Y(q1)
    assert psum2 ** 2 == cirq.PauliString(cirq.I(q0)) + 2 * cirq.X(q0) * cirq.Y(q1) + cirq.PauliString(cirq.I(q1))
    psum3 = cirq.X(q0) * cirq.Z(q1) + 1.3 * cirq.Z(q0)
    sqd = cirq.PauliSum.from_pauli_strings([2.69 * cirq.PauliString(cirq.I(q0))])
    assert cirq.approx_eq(psum3 ** 2, sqd, atol=1e-08)
    psum4 = cirq.X(q0) * cirq.Z(q1) + 1.3 * cirq.Z(q1)
    sqd2 = cirq.PauliSum.from_pauli_strings([2.69 * cirq.PauliString(cirq.I(q0)), 2.6 * cirq.X(q0)])
    assert cirq.approx_eq(psum4 ** 2, sqd2, atol=1e-08)
    for psum in [psum1, psum2, psum3, psum4]:
        assert cirq.approx_eq(psum ** 0, identity)
    psum5 = cirq.Z(q0) * cirq.Z(q1) + cirq.Z(q2) + cirq.Z(q3)
    correctresult = psum5.copy()
    for e in range(1, 9):
        assert correctresult == psum5 ** e
        correctresult *= psum5
    psum6 = cirq.X(q0) * cirq.Y(q1) + cirq.Z(q2) + cirq.X(q3)
    assert psum6 * psum6 * psum6 * psum6 * psum6 * psum6 * psum6 * psum6 == psum6 ** 8
    psum7 = cirq.X(q0) * cirq.Y(q1) + cirq.Z(q2)
    psum7copy = psum7.copy()
    assert psum7 ** 5 == psum7 * psum7 * psum7 * psum7 * psum7
    assert psum7copy == psum7