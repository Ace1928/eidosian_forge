import numpy as np
import cirq
import cirq.testing
def test_str_big():
    qs = cirq.LineQubit.range(10)
    final_simulator_state = cirq.StateVectorSimulationState(prng=np.random.RandomState(0), qubits=qs, initial_state=np.array([1] * 2 ** 10, dtype=np.complex64) * 0.03125, dtype=np.complex64)
    result = cirq.StateVectorTrialResult(cirq.ParamResolver(), {}, final_simulator_state)
    assert 'output vector: [0.03125+0.j 0.03125+0.j 0.03125+0.j ..' in str(result)