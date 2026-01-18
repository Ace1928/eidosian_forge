import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_operation_at():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    assert cirq.Moment().operation_at(a) is None
    assert cirq.Moment([cirq.X(a)]).operation_at(a) == cirq.X(a)
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).operation_at(a) == cirq.CZ(a, b)