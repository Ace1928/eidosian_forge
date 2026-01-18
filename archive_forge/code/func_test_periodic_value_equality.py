import pytest
import sympy
import cirq
def test_periodic_value_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.PeriodicValue(1, 2), cirq.PeriodicValue(1, 2), cirq.PeriodicValue(3, 2), cirq.PeriodicValue(3, 2), cirq.PeriodicValue(5, 2), cirq.PeriodicValue(-1, 2))
    eq.add_equality_group(cirq.PeriodicValue(1.5, 2.0), cirq.PeriodicValue(1.5, 2.0))
    eq.add_equality_group(cirq.PeriodicValue(0, 2))
    eq.add_equality_group(cirq.PeriodicValue(1, 3))
    eq.add_equality_group(cirq.PeriodicValue(2, 4))