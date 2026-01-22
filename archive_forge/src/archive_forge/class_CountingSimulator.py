import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class CountingSimulator(cirq.SimulatorBase[CountingStepResult, CountingTrialResult, CountingSimulationState]):

    def __init__(self, noise=None, split_untangled_states=False):
        super().__init__(noise=noise, split_untangled_states=split_untangled_states)

    def _create_partial_simulation_state(self, initial_state: Any, qubits: Sequence['cirq.Qid'], classical_data: cirq.ClassicalDataStore) -> CountingSimulationState:
        return CountingSimulationState(qubits=qubits, state=initial_state, classical_data=classical_data)

    def _create_simulator_trial_result(self, params: cirq.ParamResolver, measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[CountingSimulationState]') -> CountingTrialResult:
        return CountingTrialResult(params, measurements, final_simulator_state=final_simulator_state)

    def _create_step_result(self, sim_state: cirq.SimulationStateBase[CountingSimulationState]) -> CountingStepResult:
        return CountingStepResult(sim_state)