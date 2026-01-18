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
@pytest.mark.parametrize('method', ['line', 'barh', 'bar'])
def test_secondary_axis_font_size(self, method):
    df = DataFrame(np.random.default_rng(2).standard_normal((15, 2)), columns=list('AB')).assign(C=lambda df: df.B.cumsum()).assign(D=lambda df: df.C * 1.1)
    fontsize = 20
    sy = ['C', 'D']
    kwargs = {'secondary_y': sy, 'fontsize': fontsize, 'mark_right': True}
    ax = getattr(df.plot, method)(**kwargs)
    _check_ticks_props(axes=ax.right_ax, ylabelsize=fontsize)