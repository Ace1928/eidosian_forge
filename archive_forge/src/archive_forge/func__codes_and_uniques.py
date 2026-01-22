from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core import algorithms
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.io.formats.printing import pprint_thing
@cache_readonly
def _codes_and_uniques(self) -> tuple[npt.NDArray[np.signedinteger], ArrayLike]:
    uniques: ArrayLike
    if self._passed_categorical:
        cat = self.grouping_vector
        categories = cat.categories
        if self._observed:
            ucodes = algorithms.unique1d(cat.codes)
            ucodes = ucodes[ucodes != -1]
            if self._sort:
                ucodes = np.sort(ucodes)
        else:
            ucodes = np.arange(len(categories))
        uniques = Categorical.from_codes(codes=ucodes, categories=categories, ordered=cat.ordered, validate=False)
        codes = cat.codes
        if not self._dropna:
            na_mask = codes < 0
            if np.any(na_mask):
                if self._sort:
                    na_code = len(categories)
                    codes = np.where(na_mask, na_code, codes)
                else:
                    na_idx = na_mask.argmax()
                    na_code = algorithms.nunique_ints(codes[:na_idx])
                    codes = np.where(codes >= na_code, codes + 1, codes)
                    codes = np.where(na_mask, na_code, codes)
        if not self._observed:
            uniques = uniques.reorder_categories(self._orig_cats)
        return (codes, uniques)
    elif isinstance(self.grouping_vector, ops.BaseGrouper):
        codes = self.grouping_vector.codes_info
        uniques = self.grouping_vector.result_index._values
    elif self._uniques is not None:
        cat = Categorical(self.grouping_vector, categories=self._uniques)
        codes = cat.codes
        uniques = self._uniques
    else:
        codes, uniques = algorithms.factorize(self.grouping_vector, sort=self._sort, use_na_sentinel=self._dropna)
    return (codes, uniques)