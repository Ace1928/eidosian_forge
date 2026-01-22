from __future__ import annotations
from typing import Mapping, Union, Tuple
from collections.abc import Iterable, Mapping as _Mapping
from itertools import chain, islice
import numpy as np
from numpy.typing import ArrayLike
from qiskit.circuit import Parameter, QuantumCircuit
from .shape import ShapedMixin, ShapeInput, shape_tuple
def _standardize_shape(val: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return ``val`` or ``val[..., None]``.

    Args:
        val: The array whose shape to standardize.
        shape: The shape to standardize to.

    Returns:
        An array with one more dimension than ``len(shape)``, and whose leading dimensions match
        ``shape``.

    Raises:
        ValueError: If the leading shape of ``val`` does not match the ``shape``.
    """
    if val.shape == shape:
        val = val[..., None]
    elif val.ndim - 1 != len(shape) or val.shape[:-1] != shape:
        raise ValueError(f'Array with shape {val.shape} inconsistent with {shape}')
    return val