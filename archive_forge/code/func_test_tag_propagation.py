import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_tag_propagation():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(cirq.X(a), cirq.H(b), cirq.H(c), cirq.CZ(a, c))
    op = cirq.CircuitOperation(circuit)
    test_tag = 'test_tag'
    op = op.with_tags(test_tag)
    assert test_tag in op.tags
    sub_ops = cirq.decompose(op)
    for op in sub_ops:
        assert test_tag not in op.tags