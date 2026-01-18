import datetime
import unittest
import pytz
from wsme import utils
def test_parse_isodate(self):
    good_dates = [('2008-02-01', datetime.date(2008, 2, 1)), ('2009-01-04', datetime.date(2009, 1, 4))]
    ill_formatted_dates = ['24-12-2004']
    out_of_range_dates = ['0000-00-00', '2012-02-30']
    for s, d in good_dates:
        assert utils.parse_isodate(s) == d
    for s in ill_formatted_dates + out_of_range_dates:
        self.assertRaises(ValueError, utils.parse_isodate, s)