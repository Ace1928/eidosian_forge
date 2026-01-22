from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
@final
def _get_compressed_codes(self) -> tuple[npt.NDArray[np.signedinteger], npt.NDArray[np.intp]]:
    if len(self.groupings) > 1:
        group_index = get_group_index(self.codes, self.shape, sort=True, xnull=True)
        return compress_group_index(group_index, sort=self._sort)
    ping = self.groupings[0]
    return (ping.codes, np.arange(len(ping._group_index), dtype=np.intp))