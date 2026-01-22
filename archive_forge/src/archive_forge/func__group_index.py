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
def _group_index(self) -> Index:
    codes, uniques = self._codes_and_uniques
    if not self._dropna and self._passed_categorical:
        assert isinstance(uniques, Categorical)
        if self._sort and (codes == len(uniques)).any():
            uniques = Categorical.from_codes(np.append(uniques.codes, [-1]), uniques.categories, validate=False)
        elif len(codes) > 0:
            cat = self.grouping_vector
            na_idx = (cat.codes < 0).argmax()
            if cat.codes[na_idx] < 0:
                na_unique_idx = algorithms.nunique_ints(cat.codes[:na_idx])
                new_codes = np.insert(uniques.codes, na_unique_idx, -1)
                uniques = Categorical.from_codes(new_codes, uniques.categories, validate=False)
    return Index._with_infer(uniques, name=self.name)