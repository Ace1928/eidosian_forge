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
def test_x_string_values_ticks(self):
    df = DataFrame({'sales': [3, 2, 3], 'visits': [20, 42, 28], 'day': ['Monday', 'Tuesday', 'Wednesday']})
    ax = df.plot.area(x='day')
    ax.set_xlim(-1, 3)
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    labels_position = dict(zip(xticklabels, ax.get_xticks()))
    assert labels_position['Monday'] == 0.0
    assert labels_position['Tuesday'] == 1.0
    assert labels_position['Wednesday'] == 2.0