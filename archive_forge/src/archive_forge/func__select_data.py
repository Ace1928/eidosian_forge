from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas._typing import (
from pandas.util._validators import validate_percentile
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat
from pandas.io.formats.format import format_percentiles
def _select_data(self) -> DataFrame:
    """Select columns to be described."""
    if self.include is None and self.exclude is None:
        default_include: list[npt.DTypeLike] = [np.number, 'datetime']
        data = self.obj.select_dtypes(include=default_include)
        if len(data.columns) == 0:
            data = self.obj
    elif self.include == 'all':
        if self.exclude is not None:
            msg = "exclude must be None when include is 'all'"
            raise ValueError(msg)
        data = self.obj
    else:
        data = self.obj.select_dtypes(include=self.include, exclude=self.exclude)
    return data