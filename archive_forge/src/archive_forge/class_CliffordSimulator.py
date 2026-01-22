from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, value
from cirq.protocols import act_on
from cirq.sim import clifford, simulator_base
class CliffordSimulator(simulator_base.SimulatorBase['cirq.CliffordSimulatorStepResult', 'cirq.CliffordTrialResult', 'cirq.StabilizerChFormSimulationState']):
    """An efficient simulator for Clifford circuits."""

    def __init__(self, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None, split_untangled_states: bool=False):
        """Creates instance of `CliffordSimulator`.

        Args:
            seed: The random seed to use for this simulator.
            split_untangled_states: Optimizes simulation by running separable
                states independently and merging those states at the end.
        """
        self.init = True
        super().__init__(seed=seed, split_untangled_states=split_untangled_states)

    @staticmethod
    def is_supported_operation(op: 'cirq.Operation') -> bool:
        """Checks whether given operation can be simulated by this simulator."""
        return protocols.has_stabilizer_effect(op)

    def _create_partial_simulation_state(self, initial_state: Union[int, 'cirq.StabilizerChFormSimulationState'], qubits: Sequence['cirq.Qid'], classical_data: 'cirq.ClassicalDataStore') -> 'cirq.StabilizerChFormSimulationState':
        """Creates the StabilizerChFormSimulationState for a circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            logs: A log of the results of measurement that is added to.
            classical_data: The shared classical data container for this
                simulation.

        Returns:
            StabilizerChFormSimulationState for the circuit.
        """
        if isinstance(initial_state, clifford.StabilizerChFormSimulationState):
            return initial_state
        return clifford.StabilizerChFormSimulationState(prng=self._prng, classical_data=classical_data, qubits=qubits, initial_state=initial_state)

    def _create_step_result(self, sim_state: 'cirq.SimulationStateBase[clifford.StabilizerChFormSimulationState]'):
        return CliffordSimulatorStepResult(sim_state=sim_state)

    def _create_simulator_trial_result(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[cirq.StabilizerChFormSimulationState]'):
        return CliffordTrialResult(params=params, measurements=measurements, final_simulator_state=final_simulator_state)