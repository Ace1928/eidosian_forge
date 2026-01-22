from __future__ import annotations
from typing import TYPE_CHECKING
from pandas._libs import lib
from pandas.core.dtypes.missing import notna
from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series
def _check_is_partition(parts: Iterable, whole: Iterable):
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise ValueError('Is not a partition because intersection is not null.')
    if set.union(*parts) != whole:
        raise ValueError('Is not a partition because union is not the whole.')