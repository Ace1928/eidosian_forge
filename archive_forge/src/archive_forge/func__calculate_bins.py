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
def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
    """Calculate bins given data"""
    nd_values = data.infer_objects(copy=False)._get_numeric_data()
    values = np.ravel(nd_values)
    values = values[~isna(values)]
    hist, bins = np.histogram(values, bins=bins, range=self._bin_range)
    return bins