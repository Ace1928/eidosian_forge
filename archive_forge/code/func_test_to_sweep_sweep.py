import itertools
import pytest
import sympy
import cirq
def test_to_sweep_sweep():
    sweep = cirq.Linspace('a', 0, 1, 10)
    assert cirq.to_sweep(sweep) is sweep