import numpy as np
import pytest
import cirq
def test_product_state_equality():
    q0, q1, q2 = cirq.LineQubit.range(3)
    assert cirq.KET_PLUS(q0) == cirq.KET_PLUS(q0)
    assert cirq.KET_PLUS(q0) != cirq.KET_PLUS(q1)
    assert cirq.KET_PLUS(q0) != cirq.KET_MINUS(q0)
    assert cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1) == cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1)
    assert cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1) != cirq.KET_PLUS(q0) * cirq.KET_MINUS(q2)
    assert hash(cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1)) == hash(cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1))
    assert hash(cirq.KET_PLUS(q0) * cirq.KET_MINUS(q1)) != hash(cirq.KET_PLUS(q0) * cirq.KET_MINUS(q2))
    assert cirq.KET_PLUS(q0) != '+X(0)'