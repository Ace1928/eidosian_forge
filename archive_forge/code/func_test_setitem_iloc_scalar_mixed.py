import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
def test_setitem_iloc_scalar_mixed(self, data):
    super().test_setitem_iloc_scalar_mixed(data)