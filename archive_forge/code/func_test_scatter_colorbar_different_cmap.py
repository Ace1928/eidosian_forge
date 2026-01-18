import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_scatter_colorbar_different_cmap(self):
    df = DataFrame({'x': [1, 2, 3], 'y': [1, 3, 2], 'c': [1, 2, 3]})
    df['x2'] = df['x'] + 1
    _, ax = plt.subplots()
    df.plot('x', 'y', c='c', kind='scatter', cmap='cividis', ax=ax)
    df.plot('x2', 'y', c='c', kind='scatter', cmap='magma', ax=ax)
    assert ax.collections[0].cmap.name == 'cividis'
    assert ax.collections[1].cmap.name == 'magma'