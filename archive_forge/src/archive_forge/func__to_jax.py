import logging
from typing import Tuple, Callable
import dataclasses
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.typing import ResultBatch
def _to_jax(result: qml.typing.ResultBatch) -> qml.typing.ResultBatch:
    """Converts an arbitrary result batch to one with jax arrays.
    Args:
        result (ResultBatch): a nested structure of lists, tuples, dicts, and numpy arrays
    Returns:
        ResultBatch: a nested structure of tuples, dicts, and jax arrays
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        return tuple((_to_jax(r) for r in result))
    return jnp.array(result)