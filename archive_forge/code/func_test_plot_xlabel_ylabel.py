import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('vert', [True, False])
def test_plot_xlabel_ylabel(self, vert):
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10), 'group': np.random.default_rng(2).choice(['group1', 'group2'], 10)})
    xlabel, ylabel = ('x', 'y')
    ax = df.plot(kind='box', vert=vert, xlabel=xlabel, ylabel=ylabel)
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel