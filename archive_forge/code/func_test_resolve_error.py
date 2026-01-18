import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_error(resolve_fn):
    t = sympy.Symbol('t')
    gpt = cirq.GlobalPhaseGate(coefficient=t)
    with pytest.raises(ValueError, match='Coefficient is not unitary'):
        resolve_fn(gpt, {'t': -2})