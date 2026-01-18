import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='slice object is not iterable')
def test_setitem_slice_array(self, data):
    super().test_setitem_slice_array(data)