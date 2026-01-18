import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_value_of_calculations():
    assert not bool(cirq.ParamResolver())
    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1, 'c': 1 + 1j})
    assert bool(r)
    assert r.value_of(2 * sympy.pi) == 2 * np.pi
    assert r.value_of(4 ** sympy.Symbol('a') + sympy.Symbol('b') * 10) == 3
    assert r.value_of(sympy.I * sympy.pi) == np.pi * 1j
    assert r.value_of(sympy.Symbol('a') * 3) == 1.5
    assert r.value_of(sympy.Symbol('b') / 0.1 - sympy.Symbol('a')) == 0.5