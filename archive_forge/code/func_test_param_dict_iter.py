import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_param_dict_iter():
    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1})
    assert [key for key in r] == ['a', 'b']
    assert [r.value_of(key) for key in r] == [0.5, 0.1]
    assert list(r) == ['a', 'b']