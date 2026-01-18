import numpy as np
import pytest
import cirq
import sympy
def test_unknown_gates():
    gate = cirq.Y
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(gate(q), cirq.measure(q, key='key'))
    sim = cirq.ClassicalStateSimulator()
    with pytest.raises(ValueError):
        _ = sim.run(circuit).records