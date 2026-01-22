from __future__ import annotations
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import (
from pandas.core.algorithms import (
from pandas.core.arrays._mixins import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
@classmethod
def _from_inferred_categories(cls, inferred_categories, inferred_codes, dtype, true_values=None) -> Self:
    """
        Construct a Categorical from inferred values.

        For inferred categories (`dtype` is None) the categories are sorted.
        For explicit `dtype`, the `inferred_categories` are cast to the
        appropriate type.

        Parameters
        ----------
        inferred_categories : Index
        inferred_codes : Index
        dtype : CategoricalDtype or 'category'
        true_values : list, optional
            If none are provided, the default ones are
            "True", "TRUE", and "true."

        Returns
        -------
        Categorical
        """
    from pandas import Index, to_datetime, to_numeric, to_timedelta
    cats = Index(inferred_categories)
    known_categories = isinstance(dtype, CategoricalDtype) and dtype.categories is not None
    if known_categories:
        if is_any_real_numeric_dtype(dtype.categories.dtype):
            cats = to_numeric(inferred_categories, errors='coerce')
        elif lib.is_np_dtype(dtype.categories.dtype, 'M'):
            cats = to_datetime(inferred_categories, errors='coerce')
        elif lib.is_np_dtype(dtype.categories.dtype, 'm'):
            cats = to_timedelta(inferred_categories, errors='coerce')
        elif is_bool_dtype(dtype.categories.dtype):
            if true_values is None:
                true_values = ['True', 'TRUE', 'true']
            cats = cats.isin(true_values)
    if known_categories:
        categories = dtype.categories
        codes = recode_for_categories(inferred_codes, cats, categories)
    elif not cats.is_monotonic_increasing:
        unsorted = cats.copy()
        categories = cats.sort_values()
        codes = recode_for_categories(inferred_codes, unsorted, categories)
        dtype = CategoricalDtype(categories, ordered=False)
    else:
        dtype = CategoricalDtype(cats, ordered=False)
        codes = inferred_codes
    return cls._simple_new(codes, dtype=dtype)