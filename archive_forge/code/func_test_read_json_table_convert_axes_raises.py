import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_read_json_table_convert_axes_raises(self):
    df = DataFrame([[1, 2], [3, 4]], index=[1.0, 2.0], columns=['1.', '2.'])
    dfjson = df.to_json(orient='table')
    msg = "cannot pass both convert_axes and orient='table'"
    with pytest.raises(ValueError, match=msg):
        read_json(dfjson, orient='table', convert_axes=True)