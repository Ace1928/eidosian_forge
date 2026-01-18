import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.fixture
def times_frame():
    """Frame for testing times argument in EWM groupby."""
    return DataFrame({'A': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'], 'B': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3], 'C': to_datetime(['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-02', '2020-01-10', '2020-01-22', '2020-01-03', '2020-01-23', '2020-01-23', '2020-01-04'])})