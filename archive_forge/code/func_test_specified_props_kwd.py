import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('props, expected', [('boxprops', 'boxes'), ('whiskerprops', 'whiskers'), ('capprops', 'caps'), ('medianprops', 'medians')])
def test_specified_props_kwd(self, props, expected):
    df = DataFrame({k: np.random.default_rng(2).random(10) for k in 'ABC'})
    kwd = {props: {'color': 'C1'}}
    result = df.boxplot(return_type='dict', **kwd)
    assert result[expected][0].get_color() == 'C1'