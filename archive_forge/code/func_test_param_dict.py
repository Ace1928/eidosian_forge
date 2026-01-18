import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_param_dict():
    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1})
    r2 = cirq.ParamResolver(r)
    assert r2 is r
    assert r.param_dict == {'a': 0.5, 'b': 0.1}