import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('scalar, terms, terms_expected', ((2, {}, {}), (2, {'X': 6, 'Y': -2}, {'X': 3, 'Y': -1}), (1j, {'X': 1, 'Y': 1j}, {'X': -1j, 'Y': 1}), (-1, {'a': 10, 'b': -20}, {'a': -10, 'b': 20})))
def test_scalar_division(scalar, terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual = linear_dict / scalar
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual