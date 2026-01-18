import pytest
import cirq
def test_qubit_order_or_list():
    b = cirq.NamedQubit('b')
    implied_by_list = cirq.QubitOrder.as_qubit_order([b])
    assert implied_by_list.order_for([]) == (b,)
    implied_by_generator = cirq.QubitOrder.as_qubit_order((cirq.NamedQubit(e.name + '!') for e in [b]))
    assert implied_by_generator.order_for([]) == (cirq.NamedQubit('b!'),)
    assert implied_by_generator.order_for([]) == (cirq.NamedQubit('b!'),)
    ordered = cirq.QubitOrder.sorted_by(repr)
    passed_through = cirq.QubitOrder.as_qubit_order(ordered)
    assert ordered is passed_through