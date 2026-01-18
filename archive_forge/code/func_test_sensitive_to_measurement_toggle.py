import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_sensitive_to_measurement_toggle():
    q = cirq.NamedQubit('q')
    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(q)])]), cirq.Circuit([cirq.Moment([cirq.X(q)]), cirq.Moment([cirq.measure(q)])]))
    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(q)])]), cirq.Circuit([cirq.Moment([cirq.measure(q, invert_mask=(True,))])]))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.Circuit([cirq.Moment([cirq.measure(q)])]), cirq.Circuit([cirq.Moment([cirq.X(q)]), cirq.Moment([cirq.measure(q, invert_mask=(True,))])]))