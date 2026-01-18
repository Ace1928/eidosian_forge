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
@pytest.mark.parametrize('index_name, old_label, new_label', [(None, '', 'new'), ('old', 'old', 'new'), (None, '', ''), (None, '', 1), (None, '', [1, 2])])
@pytest.mark.parametrize('kind', ['line', 'area', 'bar'])
def test_xlabel_ylabel_dataframe_single_plot(self, kind, index_name, old_label, new_label):
    df = DataFrame([[1, 2], [2, 5]], columns=['Type A', 'Type B'])
    df.index.name = index_name
    ax = df.plot(kind=kind)
    assert ax.get_xlabel() == old_label
    assert ax.get_ylabel() == ''
    ax = df.plot(kind=kind, ylabel=new_label, xlabel=new_label)
    assert ax.get_ylabel() == str(new_label)
    assert ax.get_xlabel() == str(new_label)