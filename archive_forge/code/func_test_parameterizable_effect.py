import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable_effect(resolve_fn):
    q = cirq.NamedQubit('q')
    r = cirq.ParamResolver({'a': 0.5})
    op1 = cirq.GateOperation(cirq.Z ** sympy.Symbol('a'), [q])
    assert cirq.is_parameterized(op1)
    op2 = resolve_fn(op1, r)
    assert not cirq.is_parameterized(op2)
    assert op2 == cirq.S.on(q)