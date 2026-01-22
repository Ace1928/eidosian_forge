from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
@classmethod
def _get_atom(cls, values: ArrayLike) -> Col:
    """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
    dtype = values.dtype
    itemsize = dtype.itemsize
    shape = values.shape
    if values.ndim == 1:
        shape = (1, values.size)
    if isinstance(values, Categorical):
        codes = values.codes
        atom = cls.get_atom_data(shape, kind=codes.dtype.name)
    elif lib.is_np_dtype(dtype, 'M') or isinstance(dtype, DatetimeTZDtype):
        atom = cls.get_atom_datetime64(shape)
    elif lib.is_np_dtype(dtype, 'm'):
        atom = cls.get_atom_timedelta64(shape)
    elif is_complex_dtype(dtype):
        atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
    elif is_string_dtype(dtype):
        atom = cls.get_atom_string(shape, itemsize)
    else:
        atom = cls.get_atom_data(shape, kind=dtype.name)
    return atom