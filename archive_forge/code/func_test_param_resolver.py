import numpy as np
import pytest
import cirq
import sympy
def test_param_resolver():
    gate = cirq.CNOT ** sympy.Symbol('t')
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(gate(q0, q1))
    circuit.append(cirq.measure(q1, key='key'))
    resolver = cirq.ParamResolver({'t': 0})
    sim = cirq.ClassicalStateSimulator()
    results_with_parameter_zero = sim.run(circuit, param_resolver=resolver, repetitions=1).records
    resolver = cirq.ParamResolver({'t': 1})
    results_with_parameter_one = sim.run(circuit, param_resolver=resolver, repetitions=1).records
    np.testing.assert_equal(results_with_parameter_zero, {'key': np.array([[[0]]], dtype=np.uint8)})
    np.testing.assert_equal(results_with_parameter_one, {'key': np.array([[[1]]], dtype=np.uint8)})