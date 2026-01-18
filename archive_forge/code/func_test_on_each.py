from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_on_each():

    class CustomGate(cirq.testing.SingleQubitGate):
        pass
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = CustomGate()
    assert c.on_each() == []
    assert c.on_each(a) == [c(a)]
    assert c.on_each(a, b) == [c(a), c(b)]
    assert c.on_each(b, a) == [c(b), c(a)]
    assert c.on_each([]) == []
    assert c.on_each([a]) == [c(a)]
    assert c.on_each([a, b]) == [c(a), c(b)]
    assert c.on_each([b, a]) == [c(b), c(a)]
    assert c.on_each([a, [b, a], b]) == [c(a), c(b), c(a), c(b)]
    with pytest.raises(ValueError):
        c.on_each('abcd')
    with pytest.raises(ValueError):
        c.on_each(['abcd'])
    with pytest.raises(ValueError):
        c.on_each([a, 'abcd'])
    qubit_iterator = (q for q in [a, b, a, b])
    assert isinstance(qubit_iterator, Iterator)
    assert c.on_each(qubit_iterator) == [c(a), c(b), c(a), c(b)]