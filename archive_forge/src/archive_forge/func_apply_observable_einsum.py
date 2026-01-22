from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from .utils import (
from .apply_operation import apply_operation
def apply_observable_einsum(obs: Observable, state, is_state_batched: bool=False):
    """Applies an observable to a density matrix rho, giving obs@state.

    Args:
        obs (Operator): Operator to apply to the quantum state.
        state (array[complex]): Input quantum state.
        is_state_batched (bool): Boolean representing whether the state is batched or not.

    Returns:
        TensorLike: the result of obs@state.
    """
    num_ch_wires = len(obs.wires)
    einsum_indices = get_einsum_mapping(obs, state, _map_indices_apply_operation, is_state_batched)
    obs_mat = obs.matrix()
    obs_shape = [QUDIT_DIM] * num_ch_wires * 2
    obs_mat = math.cast(math.reshape(obs_mat, obs_shape), complex)
    return math.einsum(einsum_indices, obs_mat, state)