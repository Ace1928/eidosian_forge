import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, is_parameterized, parameter_names', [({cirq.H(cirq.LineQubit(0)): 1}, False, set()), ({cirq.X(cirq.LineQubit(0)) ** sympy.Symbol('t'): 1}, True, {'t'})])
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterized_linear_combination_of_ops(terms, is_parameterized, parameter_names, resolve_fn):
    op = cirq.LinearCombinationOfOperations(terms)
    assert cirq.is_parameterized(op) == is_parameterized
    assert cirq.parameter_names(op) == parameter_names
    resolved = resolve_fn(op, {p: 1 for p in parameter_names})
    assert not cirq.is_parameterized(resolved)