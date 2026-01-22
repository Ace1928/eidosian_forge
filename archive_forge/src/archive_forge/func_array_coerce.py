from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol, Union, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike, NDArray
def array_coerce(arr: ArrayLike | Shaped) -> NDArray | Shaped:
    """Coerce the input into an object with a shape attribute.

    Copies are avoided.

    Args:
        arr: The object to coerce.

    Returns:
        Something that is :class:`~Shaped`, and always ``numpy.ndarray`` if the input is not
        already :class:`~Shaped`.
    """
    if isinstance(arr, Shaped):
        return arr
    return np.array(arr, copy=False)