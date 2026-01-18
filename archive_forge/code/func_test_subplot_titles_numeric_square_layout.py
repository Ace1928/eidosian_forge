import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_subplot_titles_numeric_square_layout(self, iris):
    df = iris.drop('Name', axis=1).head()
    title = list(df.columns)
    plot = df.drop('SepalWidth', axis=1).plot(subplots=True, layout=(2, 2), title=title[:-1])
    title_list = [ax.get_title() for sublist in plot for ax in sublist]
    assert title_list == title[:3] + ['']