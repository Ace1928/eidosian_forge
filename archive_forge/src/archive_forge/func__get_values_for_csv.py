from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def _get_values_for_csv(self, *, na_rep: str='nan', **kwargs) -> npt.NDArray[np.object_]:
    new_levels = []
    new_codes = []
    for level, level_codes in zip(self.levels, self.codes):
        level_strs = level._get_values_for_csv(na_rep=na_rep, **kwargs)
        mask = level_codes == -1
        if mask.any():
            nan_index = len(level_strs)
            level_strs = level_strs.astype(str)
            level_strs = np.append(level_strs, na_rep)
            assert not level_codes.flags.writeable
            level_codes = level_codes.copy()
            level_codes[mask] = nan_index
        new_levels.append(level_strs)
        new_codes.append(level_codes)
    if len(new_levels) == 1:
        return Index(new_levels[0].take(new_codes[0]))._get_values_for_csv()
    else:
        mi = MultiIndex(levels=new_levels, codes=new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)
        return mi._values