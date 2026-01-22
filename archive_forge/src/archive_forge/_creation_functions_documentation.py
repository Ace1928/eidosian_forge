from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from ._dtypes import _all_dtypes
import cupy as np
from cupy.cuda import Device as _Device
from cupy_backends.cuda.api import runtime

    Array API compatible wrapper for :py:func:`np.zeros_like <numpy.zeros_like>`.

    See its docstring for more information.
    