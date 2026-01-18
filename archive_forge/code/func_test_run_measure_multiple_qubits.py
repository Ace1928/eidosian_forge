import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_run_measure_multiple_qubits():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.CliffordSimulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit()
            if b0:
                circuit.append(cirq.X(q0))
            if b1:
                circuit.append(cirq.X(q1))
            circuit.append(cirq.measure(q0, q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements, {'q(0),q(1)': [[b0, b1]] * 3})