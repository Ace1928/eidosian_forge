import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('terms_1, terms_2, terms_expected', (({}, {}, {}), ({'a': 2}, {'a': 2}, {}), ({'a': 3}, {'a': 2}, {'a': 1}), ({'X': 1}, {'Y': 2}, {'X': 1, 'Y': -2}), ({'X': 1}, {'X': 1}, {}), ({'X': 1, 'Y': 2}, {'Y': 2}, {'X': 1}), ({'X': 1, 'Y': 2}, {'Y': 3}, {'X': 1, 'Y': -1})))
def test_vector_subtraction(terms_1, terms_2, terms_expected):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    actual_1 = linear_dict_1 - linear_dict_2
    actual_2 = linear_dict_1
    actual_2 -= linear_dict_2
    expected = cirq.LinearDict(terms_expected)
    assert actual_1 == expected
    assert actual_2 == expected
    assert actual_1 == actual_2