import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_each():
    qubits = cirq.LineQubit.range(8)
    for n in range(len(qubits) + 1):
        ops = cirq.reset_each(*qubits[:n])
        assert len(ops) == n
        for i, op in enumerate(ops):
            assert isinstance(op.gate, cirq.ResetChannel)
            assert op.qubits == (qubits[i],)