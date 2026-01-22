import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
def _execute_wrapper(params, tapes, execute_fn, jpc) -> ResultBatch:
    """Executes ``tapes`` with ``params`` via ``execute_fn``"""
    new_tapes = set_parameters_on_copy_and_unwrap(tapes.vals, params, unwrap=False)
    return _to_jax(execute_fn(new_tapes))