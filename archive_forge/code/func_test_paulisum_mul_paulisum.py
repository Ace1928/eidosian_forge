import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_paulisum_mul_paulisum():
    q0, q1, q2 = cirq.LineQubit.range(3)
    psum1 = cirq.X(q0) + 2 * cirq.Y(q0) * cirq.Y(q1)
    psum2 = cirq.X(q0) * cirq.Y(q1) + 3 * cirq.Z(q2)
    assert psum1 * psum2 == cirq.Y(q1) + 3 * cirq.X(q0) * cirq.Z(q2) - 2j * cirq.Z(q0) + 6 * cirq.Y(q0) * cirq.Y(q1) * cirq.Z(q2)
    assert psum2 * psum1 == cirq.Y(q1) + 3 * cirq.X(q0) * cirq.Z(q2) + 2j * cirq.Z(q0) + 6 * cirq.Y(q0) * cirq.Y(q1) * cirq.Z(q2)
    psum3 = cirq.X(q1) + cirq.X(q2)
    psum1 *= psum3
    assert psum1 == cirq.X(q0) * cirq.X(q1) - 2j * cirq.Y(q0) * cirq.Z(q1) + cirq.X(q0) * cirq.X(q2) + 2 * cirq.Y(q0) * cirq.Y(q1) * cirq.X(q2)
    psum4 = cirq.X(q0) + cirq.Y(q0) + cirq.Z(q1)
    psum5 = cirq.Z(q0) + cirq.Y(q0) + cirq.PauliString(coefficient=1.2)
    assert psum4 * psum5 == -1j * cirq.Y(q0) + 1j * (cirq.X(q0) + cirq.Z(q0)) + (cirq.Z(q0) + cirq.Y(q0)) * cirq.Z(q1) + 1 + 1.2 * psum4
    assert psum5 * psum4 == 1j * cirq.Y(q0) + -1j * (cirq.X(q0) + cirq.Z(q0)) + (cirq.Z(q0) + cirq.Y(q0)) * cirq.Z(q1) + 1 + 1.2 * psum4