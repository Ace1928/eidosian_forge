import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_phased_fsim_resolve(resolve_fn):
    f = cirq.PhasedFSimGate(sympy.Symbol('a'), sympy.Symbol('b'), sympy.Symbol('c'), sympy.Symbol('d'), sympy.Symbol('e'))
    assert cirq.is_parameterized(f)
    f = resolve_fn(f, {'a': 1})
    assert f == cirq.PhasedFSimGate(1, sympy.Symbol('b'), sympy.Symbol('c'), sympy.Symbol('d'), sympy.Symbol('e'))
    assert cirq.is_parameterized(f)
    f = resolve_fn(f, {'b': 2})
    assert f == cirq.PhasedFSimGate(1, 2, sympy.Symbol('c'), sympy.Symbol('d'), sympy.Symbol('e'))
    assert cirq.is_parameterized(f)
    f = resolve_fn(f, {'c': 3})
    assert f == cirq.PhasedFSimGate(1, 2, 3, sympy.Symbol('d'), sympy.Symbol('e'))
    assert cirq.is_parameterized(f)
    f = resolve_fn(f, {'d': 4})
    assert f == cirq.PhasedFSimGate(1, 2, 3, 4, sympy.Symbol('e'))
    assert cirq.is_parameterized(f)
    f = resolve_fn(f, {'e': 5})
    assert f == cirq.PhasedFSimGate(1, 2, 3, 4, 5)
    assert not cirq.is_parameterized(f)