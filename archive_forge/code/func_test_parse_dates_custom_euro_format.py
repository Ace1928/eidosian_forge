from datetime import (
from io import StringIO
from dateutil.parser import parse as du_parse
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import parsing
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv
@xfail_pyarrow
@pytest.mark.parametrize('kwargs', [{'dayfirst': True}, {'day_first': True}])
def test_parse_dates_custom_euro_format(all_parsers, kwargs):
    parser = all_parsers
    data = 'foo,bar,baz\n31/01/2010,1,2\n01/02/2010,1,NA\n02/02/2010,1,2\n'
    if 'dayfirst' in kwargs:
        df = parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), names=['time', 'Q', 'NTU'], date_parser=lambda d: du_parse(d, **kwargs), header=0, index_col=0, parse_dates=True, na_values=['NA'])
        exp_index = Index([datetime(2010, 1, 31), datetime(2010, 2, 1), datetime(2010, 2, 2)], name='time')
        expected = DataFrame({'Q': [1, 1, 1], 'NTU': [2, np.nan, 2]}, index=exp_index, columns=['Q', 'NTU'])
        tm.assert_frame_equal(df, expected)
    else:
        msg = "got an unexpected keyword argument 'day_first'"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), names=['time', 'Q', 'NTU'], date_parser=lambda d: du_parse(d, **kwargs), skiprows=[0], index_col=0, parse_dates=True, na_values=['NA'])