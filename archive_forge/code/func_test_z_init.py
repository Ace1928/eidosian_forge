import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_z_init():
    z = cirq.ZPowGate(exponent=5)
    assert z.exponent == 5
    assert cirq.Z ** 0.5 != cirq.Z ** (-0.5)
    assert (cirq.Z ** (-1)) ** 0.5 == cirq.Z ** (-0.5)
    assert cirq.Z ** (-1) == cirq.Z