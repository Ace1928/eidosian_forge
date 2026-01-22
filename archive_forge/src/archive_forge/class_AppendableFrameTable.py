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
class AppendableFrameTable(AppendableTable):
    """support the new appendable table formats"""
    pandas_kind = 'frame_table'
    table_type = 'appendable_frame'
    ndim = 2
    obj_type: type[DataFrame | Series] = DataFrame

    @property
    def is_transposed(self) -> bool:
        return self.index_axes[0].axis == 1

    @classmethod
    def get_object(cls, obj, transposed: bool):
        """these are written transposed"""
        if transposed:
            obj = obj.T
        return obj

    def read(self, where=None, columns=None, start: int | None=None, stop: int | None=None):
        self.validate_version(where)
        if not self.infer_axes():
            return None
        result = self._read_axes(where=where, start=start, stop=stop)
        info = self.info.get(self.non_index_axes[0][0], {}) if len(self.non_index_axes) else {}
        inds = [i for i, ax in enumerate(self.axes) if ax is self.index_axes[0]]
        assert len(inds) == 1
        ind = inds[0]
        index = result[ind][0]
        frames = []
        for i, a in enumerate(self.axes):
            if a not in self.values_axes:
                continue
            index_vals, cvalues = result[i]
            if info.get('type') != 'MultiIndex':
                cols = Index(index_vals)
            else:
                cols = MultiIndex.from_tuples(index_vals)
            names = info.get('names')
            if names is not None:
                cols.set_names(names, inplace=True)
            if self.is_transposed:
                values = cvalues
                index_ = cols
                cols_ = Index(index, name=getattr(index, 'name', None))
            else:
                values = cvalues.T
                index_ = Index(index, name=getattr(index, 'name', None))
                cols_ = cols
            if values.ndim == 1 and isinstance(values, np.ndarray):
                values = values.reshape((1, values.shape[0]))
            if isinstance(values, np.ndarray):
                df = DataFrame(values.T, columns=cols_, index=index_, copy=False)
            elif isinstance(values, Index):
                df = DataFrame(values, columns=cols_, index=index_)
            else:
                df = DataFrame._from_arrays([values], columns=cols_, index=index_)
            if not (using_pyarrow_string_dtype() and values.dtype.kind == 'O'):
                assert (df.dtypes == values.dtype).all(), (df.dtypes, values.dtype)
            if using_pyarrow_string_dtype() and is_string_array(values, skipna=True):
                df = df.astype('string[pyarrow_numpy]')
            frames.append(df)
        if len(frames) == 1:
            df = frames[0]
        else:
            df = concat(frames, axis=1)
        selection = Selection(self, where=where, start=start, stop=stop)
        df = self.process_axes(df, selection=selection, columns=columns)
        return df