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
@pytest.mark.parametrize('xlabel, ylabel', [(None, None), ('X Label', None), (None, 'Y Label'), ('X Label', 'Y Label')])
@pytest.mark.parametrize('kind', ['scatter', 'hexbin'])
def test_xlabel_ylabel_dataframe_plane_plot(self, kind, xlabel, ylabel):
    xcol = 'Type A'
    ycol = 'Type B'
    df = DataFrame([[1, 2], [2, 5]], columns=[xcol, ycol])
    ax = df.plot(kind=kind, x=xcol, y=ycol, xlabel=xlabel, ylabel=ylabel)
    assert ax.get_xlabel() == (xcol if xlabel is None else xlabel)
    assert ax.get_ylabel() == (ycol if ylabel is None else ylabel)