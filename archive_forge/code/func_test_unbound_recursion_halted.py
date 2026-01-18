import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_unbound_recursion_halted():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    r = cirq.ParamResolver({a: b, b: a})
    assert r.value_of(a, recursive=False) == b
    assert r.value_of(r.value_of(a, recursive=False), recursive=False) == a
    r = cirq.ParamResolver({a: a})
    assert r.value_of(a) == a
    r = cirq.ParamResolver({a: a + 1})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)
    r = cirq.ParamResolver({a: b, b: a})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)
    r = cirq.ParamResolver({a: b, b: c, c: b})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)
    r = cirq.ParamResolver({a: b + c, b: 1, c: a})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)