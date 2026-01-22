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
@classmethod
def _replace_with_mask(cls, values: pa.Array | pa.ChunkedArray, mask: npt.NDArray[np.bool_] | bool, replacements: ArrayLike | Scalar):
    """
        Replace items selected with a mask.

        Analogous to pyarrow.compute.replace_with_mask, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        values : pa.Array or pa.ChunkedArray
        mask : npt.NDArray[np.bool_] or bool
        replacements : ArrayLike or Scalar
            Replacement value(s)

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
    if isinstance(replacements, pa.ChunkedArray):
        replacements = replacements.combine_chunks()
    if isinstance(values, pa.ChunkedArray) and pa.types.is_boolean(values.type):
        values = values.combine_chunks()
    try:
        return pc.replace_with_mask(values, mask, replacements)
    except pa.ArrowNotImplementedError:
        pass
    if isinstance(replacements, pa.Array):
        replacements = np.array(replacements, dtype=object)
    elif isinstance(replacements, pa.Scalar):
        replacements = replacements.as_py()
    result = np.array(values, dtype=object)
    result[mask] = replacements
    return pa.array(result, type=values.type, from_pandas=True)