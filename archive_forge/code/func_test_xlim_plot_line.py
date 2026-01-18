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
@pytest.mark.parametrize('kind', ['line', 'area'])
def test_xlim_plot_line(self, kind):
    df = DataFrame([2, 4], index=[1, 2])
    ax = df.plot(kind=kind)
    xlims = ax.get_xlim()
    assert xlims[0] < 1
    assert xlims[1] > 2