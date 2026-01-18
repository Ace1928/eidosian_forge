from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_putmask_with_wrong_mask(self, idx):
    msg = 'putmask: mask and data must be the same size'
    with pytest.raises(ValueError, match=msg):
        idx.putmask(np.ones(len(idx) + 1, np.bool_), 1)
    with pytest.raises(ValueError, match=msg):
        idx.putmask(np.ones(len(idx) - 1, np.bool_), 1)
    with pytest.raises(ValueError, match=msg):
        idx.putmask('foo', 1)