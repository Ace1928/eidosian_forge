from io import StringIO
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('usecols', [[0, 2, 3], [3, 0, 2]])
@pytest.mark.parametrize('names', [list('abcde'), list('acd')])
def test_usecols_with_parse_dates_and_names(all_parsers, usecols, names, request):
    s = '0,1,2014-01-01,09:00,4\n0,1,2014-01-02,10:00,4'
    parse_dates = [[1, 2]]
    parser = all_parsers
    if parser.engine == 'pyarrow' and (not (len(names) == 3 and usecols[0] == 0)):
        mark = pytest.mark.xfail(reason='Length mismatch in some cases, UserWarning in other')
        request.applymarker(mark)
    cols = {'a': [0, 0], 'c_d': [Timestamp('2014-01-01 09:00:00'), Timestamp('2014-01-02 10:00:00')]}
    expected = DataFrame(cols, columns=['c_d', 'a'])
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    with tm.assert_produces_warning((FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(s), names=names, parse_dates=parse_dates, usecols=usecols)
    tm.assert_frame_equal(result, expected)