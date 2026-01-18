import numpy as np
import pytest
import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
def test_nunique_smoke(self):
    n = DataFrame([[1, 2], [1, 2]]).set_index([0, 1]).index.nunique()
    assert n == 1