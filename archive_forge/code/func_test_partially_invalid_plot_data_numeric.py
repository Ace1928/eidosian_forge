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
@pytest.mark.parametrize('kind', list(plotting.PlotAccessor._common_kinds) + ['area'])
def test_partially_invalid_plot_data_numeric(self, kind):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), dtype=object)
    df[np.random.default_rng(2).random(df.shape[0]) > 0.5] = 'a'
    msg = 'no numeric data to plot'
    with pytest.raises(TypeError, match=msg):
        df.plot(kind=kind)