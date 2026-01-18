import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_resolve_symbol_division():
    B = sympy.Symbol('B')
    r = cirq.ParamResolver({'a': 1, 'b': B})
    resolved = r.value_of(sympy.Symbol('a') / sympy.Symbol('b'))
    assert resolved == sympy.core.power.Pow(B, -1)