from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_positional_raises(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    msg = 'reindex\\(\\) takes from 1 to 2 positional arguments but 3 were given'
    with pytest.raises(TypeError, match=msg):
        df.reindex([0, 1], ['A', 'B', 'C'])