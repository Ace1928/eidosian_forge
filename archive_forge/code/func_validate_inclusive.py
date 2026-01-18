from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
def validate_inclusive(inclusive: str | None) -> tuple[bool, bool]:
    """
    Check that the `inclusive` argument is among {"both", "neither", "left", "right"}.

    Parameters
    ----------
    inclusive : {"both", "neither", "left", "right"}

    Returns
    -------
    left_right_inclusive : tuple[bool, bool]

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_right_inclusive: tuple[bool, bool] | None = None
    if isinstance(inclusive, str):
        left_right_inclusive = {'both': (True, True), 'left': (True, False), 'right': (False, True), 'neither': (False, False)}.get(inclusive)
    if left_right_inclusive is None:
        raise ValueError("Inclusive has to be either 'both', 'neither', 'left' or 'right'")
    return left_right_inclusive