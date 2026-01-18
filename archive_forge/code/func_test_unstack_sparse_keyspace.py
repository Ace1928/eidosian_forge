from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_sparse_keyspace(self):
    NUM_ROWS = 1000
    df = DataFrame({'A': np.random.default_rng(2).integers(100, size=NUM_ROWS), 'B': np.random.default_rng(3).integers(300, size=NUM_ROWS), 'C': np.random.default_rng(4).integers(-7, 7, size=NUM_ROWS), 'D': np.random.default_rng(5).integers(-19, 19, size=NUM_ROWS), 'E': np.random.default_rng(6).integers(3000, size=NUM_ROWS), 'F': np.random.default_rng(7).standard_normal(NUM_ROWS)})
    idf = df.set_index(['A', 'B', 'C', 'D', 'E'])
    idf.unstack('E')