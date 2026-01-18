import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_rxyz_circuit_diagram():
    q = cirq.NamedQubit('q')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.rx(np.pi).on(q), cirq.rx(-np.pi).on(q), cirq.rx(-np.pi + 1e-05).on(q), cirq.rx(-np.pi - 1e-05).on(q), cirq.rx(3 * np.pi).on(q), cirq.rx(7 * np.pi / 2).on(q), cirq.rx(9 * np.pi / 2 + 1e-05).on(q)), '\nq: ───Rx(π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-0.5π)───Rx(0.5π)───\n    ')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.rx(np.pi).on(q), cirq.rx(np.pi / 2).on(q), cirq.rx(-np.pi + 1e-05).on(q), cirq.rx(-np.pi - 1e-05).on(q)), '\nq: ---Rx(pi)---Rx(0.5pi)---Rx(-pi)---Rx(-pi)---\n        ', use_unicode_characters=False)
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.ry(np.pi).on(q), cirq.ry(-np.pi).on(q), cirq.ry(3 * np.pi).on(q), cirq.ry(9 * np.pi / 2).on(q)), '\nq: ───Ry(π)───Ry(-π)───Ry(-π)───Ry(0.5π)───\n    ')
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.rz(np.pi).on(q), cirq.rz(-np.pi).on(q), cirq.rz(3 * np.pi).on(q), cirq.rz(9 * np.pi / 2).on(q), cirq.rz(9 * np.pi / 2 + 1e-05).on(q)), '\nq: ───Rz(π)───Rz(-π)───Rz(-π)───Rz(0.5π)───Rz(0.5π)───\n    ')