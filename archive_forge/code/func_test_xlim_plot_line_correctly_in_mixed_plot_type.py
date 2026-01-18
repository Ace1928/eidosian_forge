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
def test_xlim_plot_line_correctly_in_mixed_plot_type(self):
    fig, ax = mpl.pyplot.subplots()
    indexes = ['k1', 'k2', 'k3', 'k4']
    df = DataFrame({'s1': [1000, 2000, 1500, 2000], 's2': [900, 1400, 2000, 3000], 's3': [1500, 1500, 1600, 1200], 'secondary_y': [1, 3, 4, 3]}, index=indexes)
    df[['s1', 's2', 's3']].plot.bar(ax=ax, stacked=False)
    df[['secondary_y']].plot(ax=ax, secondary_y=True)
    xlims = ax.get_xlim()
    assert xlims[0] < 0
    assert xlims[1] > 3
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xticklabels == indexes