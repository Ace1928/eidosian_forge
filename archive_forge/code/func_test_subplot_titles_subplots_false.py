import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_subplot_titles_subplots_false(self, iris):
    df = iris.drop('Name', axis=1).head()
    title = list(df.columns)
    msg = 'Using `title` of type `list` is not supported unless `subplots=True` is passed'
    with pytest.raises(ValueError, match=msg):
        df.plot(subplots=False, title=title)