import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
@pytest.mark.parametrize('setter', ['loc', None])
def test_setitem_mask_broadcast(self, data, setter):
    super().test_setitem_mask_broadcast(data, setter)