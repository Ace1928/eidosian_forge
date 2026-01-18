import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
def test_to_int_index(self):
    index = IntIndex(10, [2, 3, 4, 5, 6])
    assert index.to_int_index() is index