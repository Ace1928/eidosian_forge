import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
class TestGroupVarFloat32(GroupVarTestMixin):
    __test__ = True
    algo = staticmethod(group_var)
    dtype = np.float32
    rtol = 0.01