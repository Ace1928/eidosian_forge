from __future__ import annotations
from collections.abc import Iterable
from typing import (
import numpy as np
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
class GroupByNthSelector:
    """
    Dynamically substituted for GroupBy.nth to enable both call and index
    """

    def __init__(self, groupby_object: groupby.GroupBy) -> None:
        self.groupby_object = groupby_object

    def __call__(self, n: PositionalIndexer | tuple, dropna: Literal['any', 'all', None]=None) -> DataFrame | Series:
        return self.groupby_object._nth(n, dropna)

    def __getitem__(self, n: PositionalIndexer | tuple) -> DataFrame | Series:
        return self.groupby_object._nth(n)