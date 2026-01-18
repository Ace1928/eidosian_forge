import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_subplot_titles_too_much(self, iris):
    df = iris.drop('Name', axis=1).head()
    title = list(df.columns)
    msg = 'The length of `title` must equal the number of columns if using `title` of type `list` and `subplots=True`'
    with pytest.raises(ValueError, match=msg):
        df.plot(subplots=True, title=title + ['kittens > puppies'])