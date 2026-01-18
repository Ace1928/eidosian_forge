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
@pytest.mark.parametrize('props, expected', [('boxprops', 'boxes'), ('whiskerprops', 'whiskers'), ('capprops', 'caps'), ('medianprops', 'medians')])
def test_specified_props_kwd_plot_box(self, props, expected):
    df = DataFrame({k: np.random.default_rng(2).random(100) for k in 'ABC'})
    kwd = {props: {'color': 'C1'}}
    result = df.plot.box(return_type='dict', **kwd)
    assert result[expected][0].get_color() == 'C1'