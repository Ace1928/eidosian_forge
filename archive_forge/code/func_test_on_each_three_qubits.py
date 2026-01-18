from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_on_each_three_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    g = cirq.testing.ThreeQubitGate()
    assert g.on_each([]) == []
    assert g.on_each([(a, b, c)]) == [g(a, b, c)]
    assert g.on_each([[a, b, c]]) == [g(a, b, c)]
    assert g.on_each([(c, b, a)]) == [g(c, b, a)]
    assert g.on_each([(a, b, c), (c, b, a)]) == [g(a, b, c), g(c, b, a)]
    assert g.on_each(zip([a, c], [b, b], [c, a])) == [g(a, b, c), g(c, b, a)]
    assert g.on_each() == []
    assert g.on_each((c, b, a)) == [g(c, b, a)]
    assert g.on_each((a, b, c), (c, b, a)) == [g(a, b, c), g(c, b, a)]
    assert g.on_each(*zip([a, c], [b, b], [c, a])) == [g(a, b, c), g(c, b, a)]
    with pytest.raises(TypeError, match='object is not iterable'):
        g.on_each(a)
    with pytest.raises(ValueError, match='Inputs to multi-qubit gates must be Sequence'):
        g.on_each(a, b, c)
    with pytest.raises(ValueError, match='Inputs to multi-qubit gates must be Sequence'):
        g.on_each([12])
    with pytest.raises(ValueError, match='Inputs to multi-qubit gates must be Sequence'):
        g.on_each([(a, b, c), 12])
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        g.on_each([(a, b, c), [(a, b, c)]])
    with pytest.raises(ValueError, match='Expected 3 qubits'):
        g.on_each([(a,)])
    with pytest.raises(ValueError, match='Expected 3 qubits'):
        g.on_each([(a, b)])
    with pytest.raises(ValueError, match='Expected 3 qubits'):
        g.on_each([(a, b, c, a)])
    with pytest.raises(ValueError, match='Expected 3 qubits'):
        g.on_each(zip([a, a], [b, b]))
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        g.on_each('abc')
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        g.on_each(('abc',))
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        g.on_each([('abc',)])
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        g.on_each([(a, 'abc')])
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        g.on_each([(a, 'bc')])
    qubit_iterator = (qs for qs in [[a, b, c], [a, b, c]])
    assert isinstance(qubit_iterator, Iterator)
    assert g.on_each(qubit_iterator) == [g(a, b, c), g(a, b, c)]