import numpy as np
import pennylane as qml
from pennylane.devices import DefaultQubitLegacy
from pennylane.pulse import ParametrizedEvolution
from pennylane.typing import TensorLike
@staticmethod
def _count_unbinned_samples(indices, batch_size, dim):
    """Count the occurences of sampled indices and convert them to relative
        counts in order to estimate their occurence probability."""
    shape = (dim + 1,) if batch_size is None else (batch_size, dim + 1)
    prob = qml.math.convert_like(jnp.zeros(shape, dtype=jnp.float64), indices)
    if batch_size is None:
        basis_states, counts = jnp.unique(indices, return_counts=True, size=dim, fill_value=-1)
        for state, count in zip(basis_states, counts):
            prob = prob.at[state].set(count / len(indices))
        return prob[:-1]
    for i, idx in enumerate(indices):
        basis_states, counts = jnp.unique(idx, return_counts=True, size=dim, fill_value=-1)
        for state, count in zip(basis_states, counts):
            prob = prob.at[i, state].set(count / len(idx))
    return prob[:, :-1]