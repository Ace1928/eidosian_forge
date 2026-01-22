from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray
def _get_res(val, ipat, imaxcount=maxcount, dtype=self._obj.dtype):
    if ipat.groups == 0:
        raise ValueError('No capture groups found in pattern.')
    matches = ipat.findall(val)
    res = np.zeros([maxcount, ipat.groups], dtype)
    if ipat.groups == 1:
        for imatch, match in enumerate(matches):
            res[imatch, 0] = match
    else:
        for imatch, match in enumerate(matches):
            for jmatch, submatch in enumerate(match):
                res[imatch, jmatch] = submatch
    return res