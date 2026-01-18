import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_operates_on_single_qubit():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    assert not cirq.Moment().operates_on_single_qubit(a)
    assert not cirq.Moment().operates_on_single_qubit(b)
    assert cirq.Moment([cirq.X(a)]).operates_on_single_qubit(a)
    assert not cirq.Moment([cirq.X(a)]).operates_on_single_qubit(b)
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on_single_qubit(a)
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on_single_qubit(b)
    assert not cirq.Moment([cirq.CZ(a, b)]).operates_on_single_qubit(c)
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on_single_qubit(a)
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on_single_qubit(b)
    assert not cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on_single_qubit(c)