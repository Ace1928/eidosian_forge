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
def test_variable_width_unicode():
    data = '\nשלום שלום\nום   שלל\nשל   ום\n'.strip('\r\n')
    encoding = 'utf8'
    kwargs = {'header': None, 'encoding': encoding}
    expected = read_fwf(BytesIO(data.encode(encoding)), colspecs=[(0, 4), (5, 9)], **kwargs)
    result = read_fwf(BytesIO(data.encode(encoding)), **kwargs)
    tm.assert_frame_equal(result, expected)