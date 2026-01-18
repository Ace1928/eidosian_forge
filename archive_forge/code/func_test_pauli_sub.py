import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_pauli_sub():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum = pstr1 - pstr2
    psum2 = cirq.PauliSum.from_pauli_strings([pstr1, -1 * pstr2])
    assert psum == psum2