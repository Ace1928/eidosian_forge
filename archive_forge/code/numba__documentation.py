from __future__ import annotations
import types
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError

    If user function is not jitted already, mark the user's function
    as jitable.

    Parameters
    ----------
    func : function
        user defined function

    Returns
    -------
    function
        Numba JITed function, or function marked as JITable by numba
    