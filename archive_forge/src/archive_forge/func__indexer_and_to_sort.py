from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
@cache_readonly
def _indexer_and_to_sort(self) -> tuple[npt.NDArray[np.intp], list[np.ndarray]]:
    v = self.level
    codes = list(self.index.codes)
    levs = list(self.index.levels)
    to_sort = codes[:v] + codes[v + 1:] + [codes[v]]
    sizes = tuple((len(x) for x in levs[:v] + levs[v + 1:] + [levs[v]]))
    comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
    ngroups = len(obs_ids)
    indexer = get_group_index_sorter(comp_index, ngroups)
    return (indexer, to_sort)