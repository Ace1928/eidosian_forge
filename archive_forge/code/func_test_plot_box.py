import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('vert', [True, False])
def test_plot_box(self, vert):
    rng = np.random.default_rng(2)
    df1 = DataFrame(rng.integers(0, 100, size=(100, 4)), columns=list('ABCD'))
    df2 = DataFrame(rng.integers(0, 100, size=(100, 4)), columns=list('ABCD'))
    xlabel, ylabel = ('x', 'y')
    _, axs = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
    df1.plot.box(ax=axs[0], vert=vert, xlabel=xlabel, ylabel=ylabel)
    df2.plot.box(ax=axs[1], vert=vert, xlabel=xlabel, ylabel=ylabel)
    for ax in axs:
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
    mpl.pyplot.close()