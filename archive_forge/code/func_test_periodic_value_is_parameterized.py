import pytest
import sympy
import cirq
@pytest.mark.parametrize('value, is_parameterized, parameter_names', [(cirq.PeriodicValue(1.0, 3.0), False, set()), (cirq.PeriodicValue(0.0, sympy.Symbol('p')), True, {'p'}), (cirq.PeriodicValue(sympy.Symbol('v'), 3.0), True, {'v'}), (cirq.PeriodicValue(sympy.Symbol('v'), sympy.Symbol('p')), True, {'p', 'v'})])
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_periodic_value_is_parameterized(value, is_parameterized, parameter_names, resolve_fn):
    assert cirq.is_parameterized(value) == is_parameterized
    assert cirq.parameter_names(value) == parameter_names
    resolved = resolve_fn(value, {p: 1 for p in parameter_names})
    assert not cirq.is_parameterized(resolved)