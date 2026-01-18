from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('index', ['string'], indirect=True)
def test_union_identity(self, index, sort):
    first = index[5:20]
    union = first.union(first, sort=sort)
    assert (union is first) is (not sort)
    union = first.union(Index([], dtype=first.dtype), sort=sort)
    assert (union is first) is (not sort)
    union = Index([], dtype=first.dtype).union(first, sort=sort)
    assert (union is first) is (not sort)