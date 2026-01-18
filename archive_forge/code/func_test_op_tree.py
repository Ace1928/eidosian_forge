import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_op_tree():
    eq = cirq.testing.EqualsTester()
    a, b = cirq.LineQubit.range(2)
    eq.add_equality_group(cirq.Moment(), cirq.Moment([]), cirq.Moment([[], [[[]]]]))
    eq.add_equality_group(cirq.Moment(cirq.X(a)), cirq.Moment([cirq.X(a)]), cirq.Moment({cirq.X(a)}))
    eq.add_equality_group(cirq.Moment(cirq.X(a), cirq.Y(b)), cirq.Moment([cirq.X(a), cirq.Y(b)]))