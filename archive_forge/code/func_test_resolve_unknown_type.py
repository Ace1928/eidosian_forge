import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_resolve_unknown_type():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    r = cirq.ParamResolver({a: b})
    assert r.value_of(cirq.X) == cirq.X