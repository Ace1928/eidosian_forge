from datetime import datetime
import numpy.testing as npt
from statsmodels.tsa.base.datetools import date_parser, dates_from_range
def test_regex_matching_month():
    t1 = '1999m4'
    t2 = '1999:m4'
    t3 = '1999:mIV'
    t4 = '1999mIV'
    result = datetime(1999, 4, 30)
    npt.assert_equal(date_parser(t1), result)
    npt.assert_equal(date_parser(t2), result)
    npt.assert_equal(date_parser(t3), result)
    npt.assert_equal(date_parser(t4), result)