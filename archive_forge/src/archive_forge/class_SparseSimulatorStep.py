from typing import Any, Iterator, List, TYPE_CHECKING, Union, Sequence, Type, Optional
import numpy as np
from cirq import ops
from cirq.sim import simulator, state_vector, state_vector_simulator, state_vector_simulation_state
class SparseSimulatorStep(state_vector.StateVectorMixin, state_vector_simulator.StateVectorStepResult):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, sim_state: 'cirq.SimulationStateBase[cirq.StateVectorSimulationState]', dtype: Type[np.complexfloating]=np.complex64):
        """Results of a step of the simulator.

        Args:
            sim_state: The qubit:SimulationState lookup for this step.
            dtype: The `numpy.dtype` used by the simulation. One of
                `numpy.complex64` or `numpy.complex128`.
        """
        qubit_map = {q: i for i, q in enumerate(sim_state.qubits)}
        super().__init__(sim_state=sim_state, qubit_map=qubit_map)
        self._dtype = dtype
        self._state_vector: Optional[np.ndarray] = None

    def state_vector(self, copy: bool=False):
        """Return the state vector at this point in the computation.

        The state is returned in the computational basis with these basis
        states defined by the qubit_map. In particular the value in the
        qubit_map is the index of the qubit, and these are translated into
        binary vectors where the last qubit is the 1s bit of the index, the
        second-to-last is the 2s bit of the index, and so forth (i.e. big
        endian ordering).

        Example:
             qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
             Then the returned vector will have indices mapped to qubit basis
             states like the following table

                |     | QubitA | QubitB | QubitC |
                | :-: | :----: | :----: | :----: |
                |  0  |   0    |   0    |   0    |
                |  1  |   0    |   0    |   1    |
                |  2  |   0    |   1    |   0    |
                |  3  |   0    |   1    |   1    |
                |  4  |   1    |   0    |   0    |
                |  5  |   1    |   0    |   1    |
                |  6  |   1    |   1    |   0    |
                |  7  |   1    |   1    |   1    |

        Args:
            copy: If True, then the returned state is a copy of the state
                vector. If False, then the state vector is not copied,
                potentially saving memory. If one only needs to read derived
                parameters from the state vector and store then using False
                can speed up simulation by eliminating a memory copy.
        """
        if self._state_vector is None:
            self._state_vector = np.array([1])
            state = self._merged_sim_state
            if state is not None:
                vector = state.target_tensor
                size = np.prod(vector.shape, dtype=np.int64)
                self._state_vector = np.reshape(vector, size)
        return self._state_vector.copy() if copy else self._state_vector

    def __repr__(self) -> str:
        return f'cirq.SparseSimulatorStep(sim_state={self._sim_state!r}, dtype=np.{np.dtype(self._dtype)!r})'