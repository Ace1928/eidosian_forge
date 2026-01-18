from __future__ import annotations
import re
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import join_keys, match
from ..exceptions import PlotnineError, PlotnineWarning
from .facet import (
from .strips import Strips, strip
def wrap_dims(n: int, nrow: Optional[int]=None, ncol: Optional[int]=None) -> tuple[int, int]:
    """
    Wrap dimensions
    """
    if nrow is None:
        if ncol is None:
            return n_to_nrow_ncol(n)
        else:
            nrow = int(np.ceil(n / ncol))
    if ncol is None:
        ncol = int(np.ceil(n / nrow))
    if not nrow * ncol >= n:
        raise PlotnineError('Allocated fewer panels than are required. Make sure the number of rows and columns can hold all the plot panels.')
    return (nrow, ncol)