from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import index as libindex
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import (
from pandas.core.arrays.categorical import (
from pandas.core.construction import extract_array
from pandas.core.indexes.base import (
from pandas.core.indexes.extension import (
def _is_dtype_compat(self, other: Index) -> Categorical:
    """
        *this is an internal non-public method*

        provide a comparison between the dtype of self and other (coercing if
        needed)

        Parameters
        ----------
        other : Index

        Returns
        -------
        Categorical

        Raises
        ------
        TypeError if the dtypes are not compatible
        """
    if isinstance(other.dtype, CategoricalDtype):
        cat = extract_array(other)
        cat = cast(Categorical, cat)
        if not cat._categories_match_up_to_permutation(self._values):
            raise TypeError('categories must match existing categories when appending')
    elif other._is_multi:
        raise TypeError('MultiIndex is not dtype-compatible with CategoricalIndex')
    else:
        values = other
        cat = Categorical(other, dtype=self.dtype)
        other = CategoricalIndex(cat)
        if not other.isin(values).all():
            raise TypeError('cannot append a non-category item to a CategoricalIndex')
        cat = other._values
        if not ((cat == values) | isna(cat) & isna(values)).all():
            raise TypeError('categories must match existing categories when appending')
    return cat