import itertools
import pytest
import sympy
import cirq
def test_to_sweeps_none():
    assert cirq.study.to_sweeps(None) == [cirq.UnitSweep]