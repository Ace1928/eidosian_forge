import numpy as np
import cirq
import cirq.testing
def test_state_vector_trial_result_state_mixin():
    qubits = cirq.LineQubit.range(2)
    final_simulator_state = cirq.StateVectorSimulationState(qubits=qubits, initial_state=np.array([0, 1, 0, 0]))
    result = cirq.StateVectorTrialResult(params=cirq.ParamResolver({'a': 2}), measurements={'m': np.array([1, 2])}, final_simulator_state=final_simulator_state)
    rho = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho, result.density_matrix_of(qubits))
    bloch = np.array([0, 0, -1])
    np.testing.assert_array_almost_equal(bloch, result.bloch_vector_of(qubits[1]))
    assert result.dirac_notation() == '|01⟩'