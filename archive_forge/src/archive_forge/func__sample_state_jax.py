from typing import List, Union, Tuple
import numpy as np
import pennylane as qml
from pennylane.ops import Sum, Hamiltonian, SProd, Prod
from pennylane.measurements import (
from pennylane.typing import TensorLike
from .apply_operation import apply_operation
from .measure import flatten_state
def _sample_state_jax(state, shots: int, prng_key, is_state_batched: bool=False, wires=None) -> np.ndarray:
    """
    Returns a series of samples of a state for the JAX interface based on the PRNG.

    Args:
        state (array[complex]): A state vector to be sampled
        shots (int): The number of samples to take
        prng_key (jax.random.PRNGKey): A``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator.
        is_state_batched (bool): whether the state is batched or not
        wires (Sequence[int]): The wires to sample

    Returns:
        ndarray[int]: Sample values of the shape (shots, num_wires)
    """
    import jax
    import jax.numpy as jnp
    key = prng_key
    total_indices = len(state.shape) - is_state_batched
    state_wires = qml.wires.Wires(range(total_indices))
    wires_to_sample = wires or state_wires
    num_wires = len(wires_to_sample)
    basis_states = np.arange(2 ** num_wires)
    flat_state = flatten_state(state, total_indices)
    with qml.queuing.QueuingManager.stop_recording():
        probs = qml.probs(wires=wires_to_sample).process_state(flat_state, state_wires)
    if is_state_batched:
        keys = []
        for _ in state:
            key, subkey = jax.random.split(key)
            keys.append(subkey)
        samples = jnp.array([jax.random.choice(_key, basis_states, shape=(shots,), p=prob) for _key, prob in zip(keys, probs)])
    else:
        samples = jax.random.choice(key, basis_states, shape=(shots,), p=probs)
    powers_of_two = 1 << np.arange(num_wires, dtype=np.int64)[::-1]
    states_sampled_base_ten = samples[..., None] & powers_of_two
    return (states_sampled_base_ten > 0).astype(np.int64)