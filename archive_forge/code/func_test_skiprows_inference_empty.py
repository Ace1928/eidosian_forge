from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
def test_skiprows_inference_empty():
    data = '\nAA   BBB  C\n12   345  6\n78   901  2\n'.strip()
    msg = 'No rows from which to infer column width'
    with pytest.raises(EmptyDataError, match=msg):
        read_fwf(StringIO(data), skiprows=3)