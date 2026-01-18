import pytest
import sympy
import cirq
def test_slice_sweep():
    sweep = cirq.Points('a', [1, 2, 3]) * cirq.Points('b', [4, 5, 6, 7])
    first_two = sweep[:2]
    assert list(first_two.param_tuples())[0] == (('a', 1), ('b', 4))
    assert list(first_two.param_tuples())[1] == (('a', 1), ('b', 5))
    assert len(list(first_two)) == 2
    middle_three = sweep[5:8]
    assert list(middle_three.param_tuples())[0] == (('a', 2), ('b', 5))
    assert list(middle_three.param_tuples())[1] == (('a', 2), ('b', 6))
    assert list(middle_three.param_tuples())[2] == (('a', 2), ('b', 7))
    assert len(list(middle_three.param_tuples())) == 3
    odd_elems = sweep[6:1:-2]
    assert list(odd_elems.param_tuples())[2] == (('a', 1), ('b', 6))
    assert list(odd_elems.param_tuples())[1] == (('a', 2), ('b', 4))
    assert list(odd_elems.param_tuples())[0] == (('a', 2), ('b', 6))
    assert len(list(odd_elems.param_tuples())) == 3
    sweep_reversed = sweep[::-1]
    assert list(sweep) == list(reversed(list(sweep_reversed)))
    single_sweep = sweep[5:6]
    assert list(single_sweep.param_tuples())[0] == (('a', 2), ('b', 5))
    assert len(list(single_sweep.param_tuples())) == 1