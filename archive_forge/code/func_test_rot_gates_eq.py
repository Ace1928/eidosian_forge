import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_rot_gates_eq():
    eq = cirq.testing.EqualsTester()
    gates = [lambda p: cirq.CZ ** p, lambda p: cirq.X ** p, lambda p: cirq.Y ** p, lambda p: cirq.Z ** p, lambda p: cirq.CNOT ** p]
    for gate in gates:
        eq.add_equality_group(gate(3.5), gate(-0.5))
        eq.make_equality_group(lambda: gate(0))
        eq.make_equality_group(lambda: gate(0.5))
    eq.add_equality_group(cirq.XPowGate(), cirq.XPowGate(exponent=1), cirq.X)
    eq.add_equality_group(cirq.YPowGate(), cirq.YPowGate(exponent=1), cirq.Y)
    eq.add_equality_group(cirq.ZPowGate(), cirq.ZPowGate(exponent=1), cirq.Z)
    eq.add_equality_group(cirq.ZPowGate(exponent=1, global_shift=-0.5), cirq.ZPowGate(exponent=5, global_shift=-0.5))
    eq.add_equality_group(cirq.ZPowGate(exponent=3, global_shift=-0.5))
    eq.add_equality_group(cirq.ZPowGate(exponent=1, global_shift=-0.1))
    eq.add_equality_group(cirq.ZPowGate(exponent=5, global_shift=-0.1))
    eq.add_equality_group(cirq.CNotPowGate(), cirq.CXPowGate(), cirq.CNotPowGate(exponent=1), cirq.CNOT)
    eq.add_equality_group(cirq.CZPowGate(), cirq.CZPowGate(exponent=1), cirq.CZ)