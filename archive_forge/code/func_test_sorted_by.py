import pytest
import cirq
def test_sorted_by():
    a = cirq.NamedQubit('2')
    b = cirq.NamedQubit('10')
    c = cirq.NamedQubit('-5')
    q = cirq.QubitOrder.sorted_by(lambda e: -int(str(e)))
    assert q.order_for([]) == ()
    assert q.order_for([a]) == (a,)
    assert q.order_for([a, b]) == (b, a)
    assert q.order_for([a, b, c]) == (b, a, c)