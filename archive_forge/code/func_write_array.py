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
def write_array(self, key: str, obj: AnyArrayLike, items: Index | None=None) -> None:
    value = extract_array(obj, extract_numpy=True)
    if key in self.group:
        self._handle.remove_node(self.group, key)
    empty_array = value.size == 0
    transposed = False
    if isinstance(value.dtype, CategoricalDtype):
        raise NotImplementedError('Cannot store a category dtype in a HDF5 dataset that uses format="fixed". Use format="table".')
    if not empty_array:
        if hasattr(value, 'T'):
            value = value.T
            transposed = True
    atom = None
    if self._filters is not None:
        with suppress(ValueError):
            atom = _tables().Atom.from_dtype(value.dtype)
    if atom is not None:
        if not empty_array:
            ca = self._handle.create_carray(self.group, key, atom, value.shape, filters=self._filters)
            ca[:] = value
        else:
            self.write_array_empty(key, value)
    elif value.dtype.type == np.object_:
        inferred_type = lib.infer_dtype(value, skipna=False)
        if empty_array:
            pass
        elif inferred_type == 'string':
            pass
        else:
            ws = performance_doc % (inferred_type, key, items)
            warnings.warn(ws, PerformanceWarning, stacklevel=find_stack_level())
        vlarr = self._handle.create_vlarray(self.group, key, _tables().ObjectAtom())
        vlarr.append(value)
    elif lib.is_np_dtype(value.dtype, 'M'):
        self._handle.create_array(self.group, key, value.view('i8'))
        getattr(self.group, key)._v_attrs.value_type = str(value.dtype)
    elif isinstance(value.dtype, DatetimeTZDtype):
        self._handle.create_array(self.group, key, value.asi8)
        node = getattr(self.group, key)
        node._v_attrs.tz = _get_tz(value.tz)
        node._v_attrs.value_type = f'datetime64[{value.dtype.unit}]'
    elif lib.is_np_dtype(value.dtype, 'm'):
        self._handle.create_array(self.group, key, value.view('i8'))
        getattr(self.group, key)._v_attrs.value_type = 'timedelta64'
    elif empty_array:
        self.write_array_empty(key, value)
    else:
        self._handle.create_array(self.group, key, value)
    getattr(self.group, key)._v_attrs.transposed = transposed