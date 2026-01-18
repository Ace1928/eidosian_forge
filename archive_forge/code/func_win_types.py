import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
@pytest.fixture(params=['triang', 'blackman', 'hamming', 'bartlett', 'bohman', 'blackmanharris', 'nuttall', 'barthann'])
def win_types(request):
    return request.param