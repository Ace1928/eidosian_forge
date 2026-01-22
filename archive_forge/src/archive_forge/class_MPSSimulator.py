import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import quimb.tensor as qtn
from cirq import devices, protocols, qis, value
from cirq.sim import simulator_base
from cirq.sim.simulation_state import SimulationState
class MPSSimulator(simulator_base.SimulatorBase['MPSSimulatorStepResult', 'MPSTrialResult', 'MPSState']):
    """An efficient simulator for MPS circuits."""

    def __init__(self, noise: 'cirq.NOISE_MODEL_LIKE'=None, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None, simulation_options: MPSOptions=MPSOptions(), grouping: Optional[Dict['cirq.Qid', int]]=None):
        """Creates instance of `MPSSimulator`.

        Args:
            noise: A noise model to apply while simulating.
            seed: The random seed to use for this simulator.
            simulation_options: Numerical options for the simulation.
            grouping: How to group qubits together, if None all are individual.

        Raises:
            ValueError: If the noise model is not unitary or a mixture.
        """
        self.init = True
        noise_model = devices.NoiseModel.from_noise_model_like(noise)
        if not protocols.has_mixture(noise_model):
            raise ValueError(f'noise must be unitary or mixture but was {noise_model}')
        self.simulation_options = simulation_options
        self.grouping = grouping
        super().__init__(noise=noise, seed=seed)

    def _create_partial_simulation_state(self, initial_state: Union[int, 'MPSState'], qubits: Sequence['cirq.Qid'], classical_data: 'cirq.ClassicalDataStore') -> 'MPSState':
        """Creates MPSState args for simulating the Circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            classical_data: The shared classical data container for this
                simulation.

        Returns:
            MPSState args for simulating the Circuit.
        """
        if isinstance(initial_state, MPSState):
            return initial_state
        return MPSState(qubits=qubits, prng=self._prng, simulation_options=self.simulation_options, grouping=self.grouping, initial_state=initial_state, classical_data=classical_data)

    def _create_step_result(self, sim_state: 'cirq.SimulationStateBase[MPSState]'):
        return MPSSimulatorStepResult(sim_state)

    def _create_simulator_trial_result(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[MPSState]') -> 'MPSTrialResult':
        """Creates a single trial results with the measurements.

        Args:
            params: A ParamResolver for determining values of Symbols.
            measurements: A dictionary from measurement key (e.g. qubit) to the
                actual measurement array.
            final_simulator_state: The final state of the simulation.

        Returns:
            A single result.
        """
        return MPSTrialResult(params=params, measurements=measurements, final_simulator_state=final_simulator_state)