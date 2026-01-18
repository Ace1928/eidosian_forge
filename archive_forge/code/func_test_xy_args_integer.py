from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('x,y,colnames', [(0, 1, ['A', 'B']), (1, 0, [0, 1])])
def test_xy_args_integer(self, x, y, colnames):
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    df.columns = colnames
    _check_plot_works(df.plot, x=x, y=y)