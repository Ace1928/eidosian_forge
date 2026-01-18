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
def test_unstack_multiple_hierarchical(self, future_stack):
    df = DataFrame(index=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]], columns=[[0, 0, 1, 1], [0, 1, 0, 1]])
    df.index.names = ['a', 'b', 'c']
    df.columns.names = ['d', 'e']
    df.unstack(['b', 'c'])