import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_sensitive_to_measurement_but_not_measured_phase():
    q = cirq.NamedQubit('q')
    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(q)])]), cirq.Circuit())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(q)])]), cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([cirq.measure(q)])]))
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a, b)])]), cirq.Circuit([cirq.Moment([cirq.Z(a)]), cirq.Moment([cirq.measure(a, b)])]))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a)])]), cirq.Circuit([cirq.Moment([cirq.Z(a)]), cirq.Moment([cirq.measure(a)])]))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a, b)])]), cirq.Circuit([cirq.Moment([cirq.T(a), cirq.S(b)]), cirq.Moment([cirq.measure(a, b)])]))
    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a)])]), cirq.Circuit([cirq.Moment([cirq.T(a), cirq.S(b)]), cirq.Moment([cirq.measure(a)])]))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(a, b)])]), cirq.Circuit([cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.measure(a, b)])]))