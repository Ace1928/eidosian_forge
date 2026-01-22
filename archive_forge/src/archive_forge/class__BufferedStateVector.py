from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
class _BufferedStateVector(qis.QuantumStateRepresentation):
    """Contains the state vector and buffer for efficient state evolution."""

    def __init__(self, state_vector: np.ndarray, buffer: Optional[np.ndarray]=None):
        """Initializes the object with the inputs.

        This initializer creates the buffer if necessary.

        Args:
            state_vector: The state vector, must be correctly formatted. The data is not checked
                for validity here due to performance concerns.
            buffer: Optional, must be same shape as the state vector. If not provided, a buffer
                will be created automatically.
        """
        self._state_vector = state_vector
        if buffer is None:
            buffer = np.empty_like(state_vector)
        self._buffer = buffer
        self._qid_shape = state_vector.shape

    @classmethod
    def create(cls, *, initial_state: Union[np.ndarray, 'cirq.STATE_VECTOR_LIKE']=0, qid_shape: Optional[Tuple[int, ...]]=None, dtype: Optional[Type[np.complexfloating]]=None, buffer: Optional[np.ndarray]=None):
        """Initializes the object with the inputs.

        This initializer creates the buffer if necessary.

        Args:
            initial_state: The density matrix, must be correctly formatted. The data is not
                checked for validity here due to performance concerns.
            qid_shape: The shape of the density matrix, if the initial state is provided as an int.
            dtype: The dtype of the density matrix, if the initial state is provided as an int.
            buffer: Optional, must be length 3 and same shape as the density matrix. If not
                provided, a buffer will be created automatically.
        Raises:
            ValueError: If initial state is provided as integer, but qid_shape is not provided.
        """
        if not isinstance(initial_state, np.ndarray):
            if qid_shape is None:
                raise ValueError('qid_shape must be provided if initial_state is not ndarray')
            state_vector = qis.to_valid_state_vector(initial_state, len(qid_shape), qid_shape=qid_shape, dtype=dtype).reshape(qid_shape)
        else:
            if qid_shape is not None:
                state_vector = initial_state.reshape(qid_shape)
            else:
                state_vector = initial_state
            if np.may_share_memory(state_vector, initial_state):
                state_vector = state_vector.copy()
        state_vector = state_vector.astype(dtype, copy=False)
        return cls(state_vector, buffer)

    def copy(self, deep_copy_buffers: bool=True) -> '_BufferedStateVector':
        """Copies the object.

        Args:
            deep_copy_buffers: True by default, False to reuse the existing buffers.
        Returns:
            A copy of the object.
        """
        return _BufferedStateVector(state_vector=self._state_vector.copy(), buffer=self._buffer.copy() if deep_copy_buffers else self._buffer)

    def kron(self, other: '_BufferedStateVector') -> '_BufferedStateVector':
        """Creates the Kronecker product with the other state vector.

        Args:
            other: The state vector with which to kron.
        Returns:
            The Kronecker product of the two state vectors.
        """
        target_tensor = transformations.state_vector_kronecker_product(self._state_vector, other._state_vector)
        return _BufferedStateVector(state_vector=target_tensor, buffer=np.empty_like(target_tensor))

    def factor(self, axes: Sequence[int], *, validate=True, atol=1e-07) -> Tuple['_BufferedStateVector', '_BufferedStateVector']:
        """Factors a state vector into two independent state vectors.

        This function should only be called on state vectors that are known to be separable, such
        as immediately after a measurement or reset operation. It does not verify that the provided
        state vector is indeed separable, and will return nonsense results for vectors
        representing entangled states.

        Args:
            axes: The axes to factor out.
            validate: Perform a validation that the state vector factors cleanly.
            atol: The absolute tolerance for the validation.

        Returns:
            A tuple with the `(extracted, remainder)` state vectors, where `extracted` means the
            sub-state vector which corresponds to the axes requested, and with the axes in the
            requested order, and where `remainder` means the sub-state vector on the remaining
            axes, in the same order as the original state vector.
        """
        extracted_tensor, remainder_tensor = transformations.factor_state_vector(self._state_vector, axes, validate=validate, atol=atol)
        extracted = _BufferedStateVector(state_vector=extracted_tensor, buffer=np.empty_like(extracted_tensor))
        remainder = _BufferedStateVector(state_vector=remainder_tensor, buffer=np.empty_like(remainder_tensor))
        return (extracted, remainder)

    def reindex(self, axes: Sequence[int]) -> '_BufferedStateVector':
        """Transposes the axes of a state vector to a specified order.

        Args:
            axes: The desired axis order.
        Returns:
            The transposed state vector.
        """
        new_tensor = transformations.transpose_state_vector_to_axis_order(self._state_vector, axes)
        return _BufferedStateVector(state_vector=new_tensor, buffer=np.empty_like(new_tensor))

    def apply_unitary(self, action: Any, axes: Sequence[int]) -> bool:
        """Apply unitary to state.

        Args:
            action: The value with a unitary to apply.
            axes: The axes on which to apply the unitary.
        Returns:
            True if the operation succeeded.
        """
        new_target_tensor = protocols.apply_unitary(action, protocols.ApplyUnitaryArgs(target_tensor=self._state_vector, available_buffer=self._buffer, axes=axes), allow_decompose=False, default=NotImplemented)
        if new_target_tensor is NotImplemented:
            return False
        self._swap_target_tensor_for(new_target_tensor)
        return True

    def apply_mixture(self, action: Any, axes: Sequence[int], prng) -> Optional[int]:
        """Apply mixture to state.

        Args:
            action: The value with a mixture to apply.
            axes: The axes on which to apply the mixture.
            prng: The pseudo random number generator to use.
        Returns:
            The mixture index if the operation succeeded, otherwise None.
        """
        mixture = protocols.mixture(action, default=None)
        if mixture is None:
            return None
        probabilities, unitaries = zip(*mixture)
        index = prng.choice(range(len(unitaries)), p=probabilities)
        shape = protocols.qid_shape(action) * 2
        unitary = unitaries[index].astype(self._state_vector.dtype).reshape(shape)
        linalg.targeted_left_multiply(unitary, self._state_vector, axes, out=self._buffer)
        self._swap_target_tensor_for(self._buffer)
        return index

    def apply_channel(self, action: Any, axes: Sequence[int], prng) -> Optional[int]:
        """Apply channel to state.

        Args:
            action: The value with a channel to apply.
            axes: The axes on which to apply the channel.
            prng: The pseudo random number generator to use.
        Returns:
            The kraus index if the operation succeeded, otherwise None.
        """
        kraus_operators = protocols.kraus(action, default=None)
        if kraus_operators is None:
            return None

        def prepare_into_buffer(k: int):
            linalg.targeted_left_multiply(left_matrix=kraus_tensors[k], right_target=self._state_vector, target_axes=axes, out=self._buffer)
        shape = protocols.qid_shape(action)
        kraus_tensors = [e.reshape(shape * 2).astype(self._state_vector.dtype) for e in kraus_operators]
        p = prng.random()
        fallback_weight = 0.0
        fallback_weight_index = 0
        for index in range(len(kraus_tensors)):
            prepare_into_buffer(index)
            weight = float(np.linalg.norm(self._buffer) ** 2)
            if weight > fallback_weight:
                fallback_weight_index = index
                fallback_weight = weight
            p -= weight
            if p < 0:
                break
        assert weight is not None, 'No Kraus operators'
        if p >= 0 or weight == 0:
            prepare_into_buffer(fallback_weight_index)
            weight = fallback_weight
            index = fallback_weight_index
        self._buffer /= np.sqrt(weight)
        self._swap_target_tensor_for(self._buffer)
        return index

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List[int]:
        """Measures the state vector.

        Args:
            axes: The axes to measure.
            seed: The random number seed to use.
        Returns:
            The measurements in order.
        """
        bits, _ = sim.measure_state_vector(self._state_vector, axes, out=self._state_vector, qid_shape=self._qid_shape, seed=seed)
        return bits

    def sample(self, axes: Sequence[int], repetitions: int=1, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
        """Samples the state vector.

        Args:
            axes: The axes to sample.
            repetitions: The number of samples to make.
            seed: The random number seed to use.
        Returns:
            The samples in order.
        """
        return sim.sample_state_vector(self._state_vector, axes, qid_shape=self._qid_shape, repetitions=repetitions, seed=seed)

    def _swap_target_tensor_for(self, new_target_tensor: np.ndarray):
        """Gives a new state vector for the system.

        Typically, the new state vector should be `args.available_buffer` where
        `args` is this `cirq.StateVectorSimulationState` instance.

        Args:
            new_target_tensor: The new system state. Must have the same shape
                and dtype as the old system state.
        """
        if new_target_tensor is self._buffer:
            self._buffer = self._state_vector
        self._state_vector = new_target_tensor

    @property
    def supports_factor(self) -> bool:
        return True