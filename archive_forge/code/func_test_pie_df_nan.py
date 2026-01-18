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
def test_pie_df_nan(self):
    df = DataFrame(np.random.default_rng(2).random((4, 4)))
    for i in range(4):
        df.iloc[i, i] = np.nan
    _, axes = mpl.pyplot.subplots(ncols=4)
    kwargs = {'normalize': True}
    with tm.assert_produces_warning(None):
        df.plot.pie(subplots=True, ax=axes, legend=True, **kwargs)
    base_expected = ['0', '1', '2', '3']
    for i, ax in enumerate(axes):
        expected = list(base_expected)
        expected[i] = ''
        result = [x.get_text() for x in ax.texts]
        assert result == expected
        result_labels = [x.get_text() for x in ax.get_legend().get_texts()]
        expected_labels = base_expected[:i] + base_expected[i + 1:]
        assert result_labels == expected_labels