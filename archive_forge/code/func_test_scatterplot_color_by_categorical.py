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
@pytest.mark.parametrize('ordered', [True, False])
@pytest.mark.parametrize('categories', (['setosa', 'versicolor', 'virginica'], ['versicolor', 'virginica', 'setosa']))
def test_scatterplot_color_by_categorical(self, ordered, categories):
    df = DataFrame([[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]], columns=['length', 'width'])
    df['species'] = pd.Categorical(['setosa', 'setosa', 'virginica', 'virginica', 'versicolor'], ordered=ordered, categories=categories)
    ax = df.plot.scatter(x=0, y=1, c='species')
    colorbar_collection, = ax.collections
    colorbar = colorbar_collection.colorbar
    expected_ticks = np.array([0.5, 1.5, 2.5])
    result_ticks = colorbar.get_ticks()
    tm.assert_numpy_array_equal(result_ticks, expected_ticks)
    expected_boundaries = np.array([0.0, 1.0, 2.0, 3.0])
    result_boundaries = colorbar._boundaries
    tm.assert_numpy_array_equal(result_boundaries, expected_boundaries)
    expected_yticklabels = categories
    result_yticklabels = [i.get_text() for i in colorbar.ax.get_ymajorticklabels()]
    assert all((i == j for i, j in zip(result_yticklabels, expected_yticklabels)))