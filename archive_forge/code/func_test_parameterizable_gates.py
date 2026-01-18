import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable_gates(resolve_fn):
    r = cirq.ParamResolver({'a': 0.5})
    g1 = cirq.ParallelGate(cirq.Z ** sympy.Symbol('a'), 2)
    assert cirq.is_parameterized(g1)
    g2 = resolve_fn(g1, r)
    assert not cirq.is_parameterized(g2)