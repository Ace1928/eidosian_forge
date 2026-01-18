import pytest
import sympy
import cirq
def test_zip_longest_compatibility():
    sweep = cirq.Zip(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6]))
    sweep_longest = cirq.ZipLongest(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6]))
    assert tuple(sweep.param_tuples()) == tuple(sweep_longest.param_tuples())
    sweep = cirq.Zip(cirq.Points('a', [1, 3]) * cirq.Points('b', [2, 4]), cirq.Points('c', [4, 5, 6, 7]))
    sweep_longest = cirq.ZipLongest(cirq.Points('a', [1, 3]) * cirq.Points('b', [2, 4]), cirq.Points('c', [4, 5, 6, 7]))
    assert tuple(sweep.param_tuples()) == tuple(sweep_longest.param_tuples())