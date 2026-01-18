import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, is_parameterized, parameter_names', [({cirq.H: 1}, False, set()), ({cirq.X ** sympy.Symbol('t'): 1}, True, {'t'})])
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterized_linear_combination_of_gates(terms, is_parameterized, parameter_names, resolve_fn):
    gate = cirq.LinearCombinationOfGates(terms)
    assert cirq.is_parameterized(gate) == is_parameterized
    assert cirq.parameter_names(gate) == parameter_names
    resolved = resolve_fn(gate, {p: 1 for p in parameter_names})
    assert not cirq.is_parameterized(resolved)