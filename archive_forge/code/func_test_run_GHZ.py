import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_run_GHZ():
    q0, q1 = (cirq.LineQubit(0), cirq.LineQubit(1))
    simulator = cirq.CliffordSimulator()
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['q(0)'])[0] < 80
    assert sum(result.measurements['q(0)'])[0] > 20