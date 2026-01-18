import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_resolve_integer_division():
    r = cirq.ParamResolver({'a': 1, 'b': 2})
    resolved = r.value_of(sympy.Symbol('a') / sympy.Symbol('b'))
    assert resolved == 0.5