import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_nonterminal_in_subcircuit():
    a, b = cirq.LineQubit.range(2)
    fc = cirq.FrozenCircuit(cirq.H(a), cirq.measure(b, key='m1'), cirq.X(b))
    op = cirq.CircuitOperation(fc)
    c = cirq.Circuit(cirq.X(a), op)
    assert isinstance(op, cirq.CircuitOperation)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()
    op = op.with_tags('test')
    c = cirq.Circuit(cirq.X(a), op)
    assert not isinstance(op, cirq.CircuitOperation)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()