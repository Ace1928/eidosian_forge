from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from contourpy._contourpy import (
from contourpy._version import __version__
from contourpy.chunk import calc_chunk_sizes
from contourpy.convert import convert_filled, convert_lines
from contourpy.dechunk import dechunk_filled, dechunk_lines
from contourpy.enum_util import as_fill_type, as_line_type, as_z_interp
def _remove_z_mask(z: ArrayLike | np.ma.MaskedArray[Any, Any] | None) -> tuple[CoordinateArray, MaskArray | None]:
    z_array = np.ma.asarray(z, dtype=np.float64)
    z_masked = np.ma.masked_invalid(z_array, copy=False)
    if np.ma.is_masked(z_masked):
        mask = np.ma.getmask(z_masked)
    else:
        mask = None
    return (np.ma.getdata(z_masked), mask)