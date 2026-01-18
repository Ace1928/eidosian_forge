import pytest
import sympy
import cirq
def test_periodic_value_approx_eq_basic():
    assert cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.0, 2.0), atol=0.1)
    assert cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.0), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.0), atol=0.1)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.0, 2.2), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.0, 2.2), atol=0.1)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.2), atol=0.3)
    assert not cirq.approx_eq(cirq.PeriodicValue(1.0, 2.0), cirq.PeriodicValue(1.2, 2.2), atol=0.1)