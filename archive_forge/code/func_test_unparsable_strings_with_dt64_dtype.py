from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_unparsable_strings_with_dt64_dtype(self):
    vals = ['aa']
    msg = '^Unknown datetime string format, unable to parse: aa, at position 0$'
    with pytest.raises(ValueError, match=msg):
        Series(vals, dtype='datetime64[ns]')
    with pytest.raises(ValueError, match=msg):
        Series(np.array(vals, dtype=object), dtype='datetime64[ns]')