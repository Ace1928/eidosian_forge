from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_index_multiindex(self):
    d = {'t1': [2, 2.5, 3], 't2': [4, 5, 6]}
    df = DataFrame(d)
    tuples = [(0, 1), (0, 2), (1, 2)]
    df['tuples'] = tuples
    index = MultiIndex.from_tuples(df['tuples'])
    df.set_index(index)