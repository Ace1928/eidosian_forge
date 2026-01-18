import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_without_operations_touching():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    assert cirq.Moment().without_operations_touching([]) == cirq.Moment()
    assert cirq.Moment().without_operations_touching([a]) == cirq.Moment()
    assert cirq.Moment().without_operations_touching([a, b]) == cirq.Moment()
    assert cirq.Moment([cirq.X(a)]).without_operations_touching([]) == cirq.Moment([cirq.X(a)])
    assert cirq.Moment([cirq.X(a)]).without_operations_touching([a]) == cirq.Moment()
    assert cirq.Moment([cirq.X(a)]).without_operations_touching([b]) == cirq.Moment([cirq.X(a)])
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([]) == cirq.Moment([cirq.CZ(a, b)])
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([a]) == cirq.Moment()
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([b]) == cirq.Moment()
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([c]) == cirq.Moment([cirq.CZ(a, b)])
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([]) == cirq.Moment([cirq.CZ(a, b), cirq.X(c)])
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([a]) == cirq.Moment([cirq.X(c)])
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([b]) == cirq.Moment([cirq.X(c)])
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([c]) == cirq.Moment([cirq.CZ(a, b)])
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([a, b]) == cirq.Moment([cirq.X(c)])
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([a, c]) == cirq.Moment()