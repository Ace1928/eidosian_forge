import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_scatter_colors_not_raising_warnings(self):
    df = DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
    with tm.assert_produces_warning(None):
        df.plot.scatter(x='x', y='y', c='b')