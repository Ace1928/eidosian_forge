from io import StringIO
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('usecols', [[0, 2, 3], [3, 0, 2]])
def test_usecols_with_parse_dates(all_parsers, usecols):
    data = 'a,b,c,d,e\n0,1,2014-01-01,09:00,4\n0,1,2014-01-02,10:00,4'
    parser = all_parsers
    parse_dates = [[1, 2]]
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    cols = {'a': [0, 0], 'c_d': [Timestamp('2014-01-01 09:00:00'), Timestamp('2014-01-02 10:00:00')]}
    expected = DataFrame(cols, columns=['c_d', 'a'])
    if parser.engine == 'pyarrow':
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
                parser.read_csv(StringIO(data), usecols=usecols, parse_dates=parse_dates)
        return
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), usecols=usecols, parse_dates=parse_dates)
    tm.assert_frame_equal(result, expected)