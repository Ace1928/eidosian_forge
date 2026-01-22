from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]):
    if is_integer(bins):
        if self.by is not None:
            by_modified = unpack_single_str_list(self.by)
            grouped = self.data.groupby(by_modified)[self.columns]
            bins = [self._calculate_bins(group, bins) for key, group in grouped]
        else:
            bins = self._calculate_bins(self.data, bins)
    return bins