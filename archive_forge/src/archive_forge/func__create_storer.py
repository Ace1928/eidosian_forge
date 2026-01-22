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
def _create_storer(self, group, format=None, value: DataFrame | Series | None=None, encoding: str='UTF-8', errors: str='strict') -> GenericFixed | Table:
    """return a suitable class to operate"""
    cls: type[GenericFixed | Table]
    if value is not None and (not isinstance(value, (Series, DataFrame))):
        raise TypeError('value must be None, Series, or DataFrame')
    pt = _ensure_decoded(getattr(group._v_attrs, 'pandas_type', None))
    tt = _ensure_decoded(getattr(group._v_attrs, 'table_type', None))
    if pt is None:
        if value is None:
            _tables()
            assert _table_mod is not None
            if getattr(group, 'table', None) or isinstance(group, _table_mod.table.Table):
                pt = 'frame_table'
                tt = 'generic_table'
            else:
                raise TypeError('cannot create a storer if the object is not existing nor a value are passed')
        else:
            if isinstance(value, Series):
                pt = 'series'
            else:
                pt = 'frame'
            if format == 'table':
                pt += '_table'
    if 'table' not in pt:
        _STORER_MAP = {'series': SeriesFixed, 'frame': FrameFixed}
        try:
            cls = _STORER_MAP[pt]
        except KeyError as err:
            raise TypeError(f'cannot properly create the storer for: [_STORER_MAP] [group->{group},value->{type(value)},format->{format}') from err
        return cls(self, group, encoding=encoding, errors=errors)
    if tt is None:
        if value is not None:
            if pt == 'series_table':
                index = getattr(value, 'index', None)
                if index is not None:
                    if index.nlevels == 1:
                        tt = 'appendable_series'
                    elif index.nlevels > 1:
                        tt = 'appendable_multiseries'
            elif pt == 'frame_table':
                index = getattr(value, 'index', None)
                if index is not None:
                    if index.nlevels == 1:
                        tt = 'appendable_frame'
                    elif index.nlevels > 1:
                        tt = 'appendable_multiframe'
    _TABLE_MAP = {'generic_table': GenericTable, 'appendable_series': AppendableSeriesTable, 'appendable_multiseries': AppendableMultiSeriesTable, 'appendable_frame': AppendableFrameTable, 'appendable_multiframe': AppendableMultiFrameTable, 'worm': WORMTable}
    try:
        cls = _TABLE_MAP[tt]
    except KeyError as err:
        raise TypeError(f'cannot properly create the storer for: [_TABLE_MAP] [group->{group},value->{type(value)},format->{format}') from err
    return cls(self, group, encoding=encoding, errors=errors)