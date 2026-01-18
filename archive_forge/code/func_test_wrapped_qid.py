from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_wrapped_qid():
    assert type(ValidQubit('a').with_dimension(3)) is not ValidQubit
    assert type(ValidQubit('a').with_dimension(2)) is ValidQubit
    assert type(ValidQubit('a').with_dimension(5).with_dimension(2)) is ValidQubit
    assert ValidQubit('a').with_dimension(3).with_dimension(4) == ValidQubit('a').with_dimension(4)
    assert ValidQubit('a').with_dimension(3).qubit == ValidQubit('a')
    assert ValidQubit('a').with_dimension(3) == ValidQubit('a').with_dimension(3)
    assert ValidQubit('a').with_dimension(3) < ValidQubit('a').with_dimension(4)
    assert ValidQubit('a').with_dimension(3) < ValidQubit('b').with_dimension(3)
    assert ValidQubit('a').with_dimension(4) < ValidQubit('b').with_dimension(3)
    cirq.testing.assert_equivalent_repr(ValidQubit('a').with_dimension(3), global_vals={'ValidQubit': ValidQubit})
    assert str(ValidQubit('a').with_dimension(3)) == 'TQ_a (d=3)'
    assert ValidQubit('zz').with_dimension(3)._json_dict_() == {'qubit': ValidQubit('zz'), 'dimension': 3}