from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_op():
    a, b, c, d = cirq.LineQubit.range(4)
    g = ValiGate()
    op = g(a, b)
    assert op.controlled_by() is op
    controlled_op = op.controlled_by(c, d)
    assert controlled_op.sub_operation == op
    assert controlled_op.controls == (c, d)