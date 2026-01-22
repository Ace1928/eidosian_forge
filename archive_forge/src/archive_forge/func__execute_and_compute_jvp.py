import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
def _execute_and_compute_jvp(tapes, execute_fn, jpc, primals, tangents):
    """Compute the results and jvps for ``tapes`` with ``primals[0]`` parameters via
    ``jpc``.
    """
    new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, primals[0], unwrap=False)
    res, jvps = jpc.execute_and_compute_jvp(new_tapes, tangents[0])
    return (_to_jax(res), _to_jax(jvps))