import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, t.trainable_params)
    return convert_to_numpy_parameters(tc) if unwrap else tc