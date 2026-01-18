import pytest
import sympy
import cirq
def test_periodic_value_types_mismatch():
    assert not cirq.approx_eq(cirq.PeriodicValue(0.0, 2.0), 0.0, atol=0.2)
    assert not cirq.approx_eq(0.0, cirq.PeriodicValue(0.0, 2.0), atol=0.2)