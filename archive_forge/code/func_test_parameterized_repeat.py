import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_parameterized_repeat():
    q = cirq.LineQubit(0)
    op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q))) ** sympy.Symbol('a')
    assert cirq.parameter_names(op) == {'a'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 0})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 2})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': -1})
    assert np.allclose(result.state_vector(), [0, 1])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))
    op = op ** (-1)
    assert cirq.parameter_names(op) == {'a'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 0})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 2})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': -1})
    assert np.allclose(result.state_vector(), [0, 1])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))
    op = op ** sympy.Symbol('b')
    assert cirq.parameter_names(op) == {'a', 'b'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 2, 'b': 1})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 2})
    assert np.allclose(result.state_vector(), [1, 0])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5, 'b': 1})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))
    op = op ** 2.0
    assert cirq.parameter_names(op) == {'a', 'b'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 1})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5, 'b': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 1.5})
    assert np.allclose(result.state_vector(), [0, 1])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5, 'b': 1.5})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))