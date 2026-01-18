import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_key_empty(self, sort_by_key):
    df = DataFrame(np.array([]))
    df.sort_values(0, key=sort_by_key)
    df.sort_index(key=sort_by_key)