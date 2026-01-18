from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_sortlevel(self):
    index = Index([5, 4, 3, 2, 1])
    with pytest.raises(Exception, match='ascending must be a single bool value or'):
        index.sortlevel(ascending='True')
    with pytest.raises(Exception, match='ascending must be a list of bool values of length 1'):
        index.sortlevel(ascending=[True, True])
    with pytest.raises(Exception, match='ascending must be a bool value'):
        index.sortlevel(ascending=['True'])
    expected = Index([1, 2, 3, 4, 5])
    result = index.sortlevel(ascending=[True])
    tm.assert_index_equal(result[0], expected)
    expected = Index([1, 2, 3, 4, 5])
    result = index.sortlevel(ascending=True)
    tm.assert_index_equal(result[0], expected)
    expected = Index([5, 4, 3, 2, 1])
    result = index.sortlevel(ascending=False)
    tm.assert_index_equal(result[0], expected)