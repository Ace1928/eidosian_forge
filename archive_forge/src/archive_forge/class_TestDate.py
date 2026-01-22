import unittest
import time
from datetime import datetime, timedelta
from isodate import strftime
from isodate import LOCAL
from isodate import DT_EXT_COMPLETE
from isodate import tzinfo
class TestDate(unittest.TestCase):
    """
        A test case template to test ISO date formatting.
        """

    def localtime_mock(self, secs):
        """
            mock time.localtime so that it always returns a time_struct with
            tm_idst=1
            """
        tt = self.ORIG['localtime'](secs)
        if tt.tm_year < 2000:
            dst = 1
        else:
            dst = 0
        tt = (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec, tt.tm_wday, tt.tm_yday, dst)
        return time.struct_time(tt)

    def setUp(self):
        self.ORIG = {}
        self.ORIG['STDOFFSET'] = tzinfo.STDOFFSET
        self.ORIG['DSTOFFSET'] = tzinfo.DSTOFFSET
        self.ORIG['DSTDIFF'] = tzinfo.DSTDIFF
        self.ORIG['localtime'] = time.localtime
        tzinfo.STDOFFSET = timedelta(seconds=36000)
        tzinfo.DSTOFFSET = timedelta(seconds=39600)
        tzinfo.DSTDIFF = tzinfo.DSTOFFSET - tzinfo.STDOFFSET
        time.localtime = self.localtime_mock

    def tearDown(self):
        tzinfo.STDOFFSET = self.ORIG['STDOFFSET']
        tzinfo.DSTOFFSET = self.ORIG['DSTOFFSET']
        tzinfo.DSTDIFF = self.ORIG['DSTDIFF']
        time.localtime = self.ORIG['localtime']

    def test_format(self):
        """
            Take date object and create ISO string from it.
            This is the reverse test to test_parse.
            """
        if expectation is None:
            self.assertRaises(AttributeError, strftime(dt, format))
        else:
            self.assertEqual(strftime(dt, format), expectation)