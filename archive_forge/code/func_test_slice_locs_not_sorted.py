from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slice_locs_not_sorted(self):
    index = MultiIndex(levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))], codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array([0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])])
    msg = '[Kk]ey length.*greater than MultiIndex lexsort depth'
    with pytest.raises(KeyError, match=msg):
        index.slice_locs((1, 0, 1), (2, 1, 0))
    sorted_index, _ = index.sortlevel(0)
    sorted_index.slice_locs((1, 0, 1), (2, 1, 0))