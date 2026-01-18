from __future__ import annotations
import functools
import operator
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
from pandas.core.strings.base import BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset
def value_counts(self, dropna: bool=True) -> Series:
    """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
    pa_type = self._pa_array.type
    if pa_version_under11p0 and pa.types.is_duration(pa_type):
        data = self._pa_array.cast(pa.int64())
    else:
        data = self._pa_array
    from pandas import Index, Series
    vc = data.value_counts()
    values = vc.field(0)
    counts = vc.field(1)
    if dropna and data.null_count > 0:
        mask = values.is_valid()
        values = values.filter(mask)
        counts = counts.filter(mask)
    if pa_version_under11p0 and pa.types.is_duration(pa_type):
        values = values.cast(pa_type)
    counts = ArrowExtensionArray(counts)
    index = Index(type(self)(values))
    return Series(counts, index=index, name='count', copy=False)