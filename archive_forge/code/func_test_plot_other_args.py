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
@pytest.mark.parametrize('kwargs', [{'yticks': [1, 5, 10]}, {'xticks': [1, 5, 10]}, {'ylim': (-100, 100), 'xlim': (-100, 100)}, {'default_axes': True, 'subplots': True, 'title': 'blah'}])
def test_plot_other_args(self, kwargs):
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
    _check_plot_works(df.plot, **kwargs)