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
@pytest.mark.parametrize('date_string,dayfirst,expected', [('13/02/2019', True, datetime(2019, 2, 13)), ('02/13/2019', False, datetime(2019, 2, 13)), ('04/02/2019', True, datetime(2019, 2, 4))])
def test_parse_delimited_date_swap_no_warning(all_parsers, date_string, dayfirst, expected, request):
    parser = all_parsers
    expected = DataFrame({0: [expected]}, dtype='datetime64[ns]')
    if parser.engine == 'pyarrow':
        if not dayfirst:
            pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
        msg = "The 'dayfirst' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
        return
    result = parser.read_csv(StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
    tm.assert_frame_equal(result, expected)