import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_z_pushes_past_xy_and_phases_it():
    q = cirq.NamedQubit('q')
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(q) ** 0.5]), cirq.Moment([cirq.Y(q) ** 0.25])]), expected=cirq.Circuit([cirq.Moment(), cirq.Moment([cirq.X(q) ** 0.25]), cirq.Moment([cirq.Z(q) ** 0.5])]))