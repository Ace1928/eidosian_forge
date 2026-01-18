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
def write_data(self, chunksize: int | None, dropna: bool=False) -> None:
    """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
    names = self.dtype.names
    nrows = self.nrows_expected
    masks = []
    if dropna:
        for a in self.values_axes:
            mask = isna(a.data).all(axis=0)
            if isinstance(mask, np.ndarray):
                masks.append(mask.astype('u1', copy=False))
    if len(masks):
        mask = masks[0]
        for m in masks[1:]:
            mask = mask & m
        mask = mask.ravel()
    else:
        mask = None
    indexes = [a.cvalues for a in self.index_axes]
    nindexes = len(indexes)
    assert nindexes == 1, nindexes
    values = [a.take_data() for a in self.values_axes]
    values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in values]
    bvalues = []
    for i, v in enumerate(values):
        new_shape = (nrows,) + self.dtype[names[nindexes + i]].shape
        bvalues.append(v.reshape(new_shape))
    if chunksize is None:
        chunksize = 100000
    rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
    chunks = nrows // chunksize + 1
    for i in range(chunks):
        start_i = i * chunksize
        end_i = min((i + 1) * chunksize, nrows)
        if start_i >= end_i:
            break
        self.write_data_chunk(rows, indexes=[a[start_i:end_i] for a in indexes], mask=mask[start_i:end_i] if mask is not None else None, values=[v[start_i:end_i] for v in bvalues])