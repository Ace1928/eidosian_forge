import abc
from typing import Any, Dict, Iterator, Sequence, Type, TYPE_CHECKING, Generic, TypeVar
import numpy as np
from cirq import _compat, ops, value, qis
from cirq.sim import simulator, state_vector, simulator_base
from cirq.protocols import qid_shape
class SimulatesIntermediateStateVector(Generic[TStateVectorStepResult], simulator_base.SimulatorBase[TStateVectorStepResult, 'cirq.StateVectorTrialResult', 'cirq.StateVectorSimulationState'], simulator.SimulatesAmplitudes, metaclass=abc.ABCMeta):
    """A simulator that accesses its state vector as it does its simulation.

    Implementors of this interface should implement the _core_iterator
    method."""

    def __init__(self, *, dtype: Type[np.complexfloating]=np.complex64, noise: 'cirq.NOISE_MODEL_LIKE'=None, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None, split_untangled_states: bool=False):
        super().__init__(dtype=dtype, noise=noise, seed=seed, split_untangled_states=split_untangled_states)

    def _create_simulator_trial_result(self, params: 'cirq.ParamResolver', measurements: Dict[str, np.ndarray], final_simulator_state: 'cirq.SimulationStateBase[cirq.StateVectorSimulationState]') -> 'cirq.StateVectorTrialResult':
        return StateVectorTrialResult(params=params, measurements=measurements, final_simulator_state=final_simulator_state)

    def compute_amplitudes_sweep_iter(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Iterator[Sequence[complex]]:
        if isinstance(bitstrings, np.ndarray) and len(bitstrings.shape) > 1:
            raise ValueError(f'The list of bitstrings must be input as a 1-dimensional array of ints. Got an array with shape {bitstrings.shape}.')
        if isinstance(bitstrings, tuple):
            bitstrings = list(bitstrings)
        trial_result_iter = self.simulate_sweep_iter(program, params, qubit_order)
        yield from (trial_result.final_state_vector[bitstrings].tolist() for trial_result in trial_result_iter)