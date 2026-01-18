import numpy as np
import pytest
import sympy
import cirq
def test_trace_distance():
    foo = sympy.Symbol('foo')
    assert cirq.trace_distance_bound(cirq.XX ** foo) == 1.0
    assert cirq.trace_distance_bound(cirq.YY ** foo) == 1.0
    assert cirq.trace_distance_bound(cirq.ZZ ** foo) == 1.0
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.XX), 1.0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.YY ** 0), 0)
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.ZZ ** (1 / 3)), np.sin(np.pi / 6))