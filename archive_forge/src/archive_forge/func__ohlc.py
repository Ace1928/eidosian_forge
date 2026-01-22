import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def _ohlc(group):
    if isna(group).all():
        return np.repeat(np.nan, 4)
    return [group[0], group.max(), group.min(), group[-1]]