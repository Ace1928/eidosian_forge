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
def test_plot_no_rows(self):
    df = DataFrame(columns=['foo'], dtype=int)
    assert df.empty
    ax = df.plot()
    assert len(ax.get_lines()) == 1
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == 0
    assert len(line.get_ydata()) == 0