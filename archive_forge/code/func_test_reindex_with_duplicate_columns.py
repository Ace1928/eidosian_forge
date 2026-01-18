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
def test_reindex_with_duplicate_columns(self):
    df = DataFrame([[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=['bar', 'a', 'a'])
    msg = 'cannot reindex on an axis with duplicate labels'
    with pytest.raises(ValueError, match=msg):
        df.reindex(columns=['bar'])
    with pytest.raises(ValueError, match=msg):
        df.reindex(columns=['bar', 'foo'])