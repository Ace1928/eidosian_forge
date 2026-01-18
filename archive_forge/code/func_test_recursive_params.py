import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_recursive_params():
    q = cirq.LineQubit(0)
    a, a2, b, b2 = sympy.symbols('a a2 b b2')
    circuitop = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q) ** a, cirq.Z(q) ** b), param_resolver=cirq.ParamResolver({a: b, b: a}))
    outer_params = {a: a2, a2: 0, b: b2, b2: 1}
    resolved = cirq.resolve_parameters(circuitop, outer_params)
    assert resolved.param_resolver.param_dict == {a: 1, b: 0}
    resolved = cirq.resolve_parameters(circuitop, outer_params, recursive=False)
    assert resolved.param_resolver.param_dict == {a: b2, b: a2}
    with pytest.raises(RecursionError):
        cirq.resolve_parameters(circuitop, {a: a2, a2: a})
    resolved = cirq.resolve_parameters(circuitop, {a: b, b: a}, recursive=False)
    assert resolved.param_resolver.param_dict == {}
    result = cirq.Simulator().simulate(cirq.Circuit(circuitop), param_resolver=outer_params)
    assert np.allclose(result.state_vector(), [0, 1])