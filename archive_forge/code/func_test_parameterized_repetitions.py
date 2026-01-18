from typing import List, Sequence, Tuple
import numpy as np
import sympy
import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator
def test_parameterized_repetitions():
    q = cirq.LineQid(0, dimension=5)
    x = cirq.XPowGate(dimension=5)
    circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(x(q), cirq.measure(q, key='a')), repetitions=sympy.Symbol('r'), use_repetition_ids=False))
    sim = CustomStateSimulator(ComputationalBasisSimState)
    r = sim.run_sweep(circuit, [{'r': i} for i in range(1, 5)])
    assert np.allclose(r[0].records['a'], np.array([[1]]))
    assert np.allclose(r[1].records['a'], np.array([[1], [2]]))
    assert np.allclose(r[2].records['a'], np.array([[1], [2], [3]]))
    assert np.allclose(r[3].records['a'], np.array([[1], [2], [3], [4]]))