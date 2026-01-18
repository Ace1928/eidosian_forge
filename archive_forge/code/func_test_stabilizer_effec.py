import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_stabilizer_effec():
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.X)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.H)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.S)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.CNOT)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.CZ)
    qubits = cirq.LineQubit.range(2)
    gate = cirq.CliffordGate.from_op_list([cirq.H(qubits[1]), cirq.CZ(*qubits), cirq.H(qubits[1])], qubits)
    assert cirq.has_stabilizer_effect(gate)