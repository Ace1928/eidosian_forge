import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_unphaseable_causes_earlier_merge_without_size_increase():

    class UnknownGate(cirq.testing.SingleQubitGate):
        pass
    u = UnknownGate()
    q = cirq.NamedQubit('q')
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([u(q)]), cirq.Moment([cirq.Z(q) ** 0.5]), cirq.Moment([cirq.X(q)]), cirq.Moment([cirq.Z(q) ** 0.25]), cirq.Moment([cirq.X(q)]), cirq.Moment([u(q)])]), expected=cirq.Circuit([cirq.Moment([cirq.Z(q)]), cirq.Moment([u(q)]), cirq.Moment(), cirq.Moment([cirq.PhasedXPowGate(phase_exponent=-0.5)(q)]), cirq.Moment(), cirq.Moment([cirq.PhasedXPowGate(phase_exponent=-0.75).on(q)]), cirq.Moment([cirq.Z(q) ** 0.75]), cirq.Moment([u(q)])]))