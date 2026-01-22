from __future__ import annotations
from typing import Union
import numpy as np

    Makes a 2x2 matrix that corresponds to Z-rotation gate.
    This is a fast implementation that does not allocate the output matrix.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    