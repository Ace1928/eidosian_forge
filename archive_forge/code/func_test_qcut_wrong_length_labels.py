import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('labels', [['a', 'b', 'c'], list(range(3))])
def test_qcut_wrong_length_labels(labels):
    values = range(10)
    msg = 'Bin labels must be one fewer than the number of bin edges'
    with pytest.raises(ValueError, match=msg):
        qcut(values, 4, labels=labels)