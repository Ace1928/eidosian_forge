from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _is_valid_na_for(self, dtype: DtypeObj) -> bool:
    """
        Check that we are all-NA of a type/dtype that is compatible with this dtype.
        Augments `self.is_na` with an additional check of the type of NA values.
        """
    if not self.is_na:
        return False
    blk = self.block
    if blk.dtype.kind == 'V':
        return True
    if blk.dtype == object:
        values = blk.values
        return all((is_valid_na_for_dtype(x, dtype) for x in values.ravel(order='K')))
    na_value = blk.fill_value
    if na_value is NaT and blk.dtype != dtype:
        return False
    if na_value is NA and needs_i8_conversion(dtype):
        return False
    return is_valid_na_for_dtype(na_value, dtype)