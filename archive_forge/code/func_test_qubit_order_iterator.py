import pytest
import cirq
def test_qubit_order_iterator():
    generator = (q for q in cirq.LineQubit.range(5))
    assert cirq.QubitOrder.explicit(generator).order_for((cirq.LineQubit(3),)) == tuple(cirq.LineQubit.range(5))
    generator = (q for q in cirq.LineQubit.range(5))
    assert cirq.QubitOrder.as_qubit_order(generator).order_for((cirq.LineQubit(3),)) == tuple(cirq.LineQubit.range(5))