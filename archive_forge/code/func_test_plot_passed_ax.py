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
@pytest.mark.slow
def test_plot_passed_ax(self):
    df = DataFrame({'x': np.random.default_rng(2).random(10)})
    _, ax = mpl.pyplot.subplots()
    axes = df.plot.bar(subplots=True, ax=ax)
    assert len(axes) == 1
    result = ax.axes
    assert result is axes[0]