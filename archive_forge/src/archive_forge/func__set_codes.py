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
def _set_codes(self, codes, *, level=None, copy: bool=False, validate: bool=True, verify_integrity: bool=False) -> None:
    if validate:
        if level is None and len(codes) != self.nlevels:
            raise ValueError('Length of codes must match number of levels')
        if level is not None and len(codes) != len(level):
            raise ValueError('Length of codes must match length of levels.')
    level_numbers: list[int] | range
    if level is None:
        new_codes = FrozenList((_coerce_indexer_frozen(level_codes, lev, copy=copy).view() for lev, level_codes in zip(self._levels, codes)))
        level_numbers = range(len(new_codes))
    else:
        level_numbers = [self._get_level_number(lev) for lev in level]
        new_codes_list = list(self._codes)
        for lev_num, level_codes in zip(level_numbers, codes):
            lev = self.levels[lev_num]
            new_codes_list[lev_num] = _coerce_indexer_frozen(level_codes, lev, copy=copy)
        new_codes = FrozenList(new_codes_list)
    if verify_integrity:
        new_codes = self._verify_integrity(codes=new_codes, levels_to_verify=level_numbers)
    self._codes = new_codes
    self._reset_cache()