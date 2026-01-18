import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_runtime_types_of_rot_gates():
    for gate_type in [lambda p: cirq.CZPowGate(exponent=p), lambda p: cirq.XPowGate(exponent=p), lambda p: cirq.YPowGate(exponent=p), lambda p: cirq.ZPowGate(exponent=p)]:
        p = gate_type(sympy.Symbol('a'))
        assert cirq.unitary(p, None) is None
        assert cirq.pow(p, 2, None) == gate_type(2 * sympy.Symbol('a'))
        assert cirq.inverse(p, None) == gate_type(-sympy.Symbol('a'))
        c = gate_type(0.5)
        assert cirq.unitary(c, None) is not None
        assert cirq.pow(c, 2) == gate_type(1)
        assert cirq.inverse(c) == gate_type(-0.5)