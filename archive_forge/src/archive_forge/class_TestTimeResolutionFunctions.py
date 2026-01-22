import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
class TestTimeResolutionFunctions(unittest.TestCase):

    def test_get_time_resolution(self):
        self.assertEqual(get_time_resolution('01:23:45'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('24:00:00'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('23:21:28,512400'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('23:21:28.512400'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('01:23'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('24:00'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('01:23,4567'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('01:23.4567'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('012345'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('240000'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('0123'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('2400'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('01'), TimeResolution.Hours)
        self.assertEqual(get_time_resolution('24'), TimeResolution.Hours)
        self.assertEqual(get_time_resolution('12,5'), TimeResolution.Hours)
        self.assertEqual(get_time_resolution('12.5'), TimeResolution.Hours)
        self.assertEqual(get_time_resolution('232128.512400+00:00'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('0123.4567+00:00'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('01.4567+00:00'), TimeResolution.Hours)
        self.assertEqual(get_time_resolution('01:23:45+00:00'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('24:00:00+00:00'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('23:21:28.512400+00:00'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('01:23+00:00'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('24:00+00:00'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('01:23.4567+00:00'), TimeResolution.Minutes)
        self.assertEqual(get_time_resolution('23:21:28.512400+11:15'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('23:21:28.512400-12:34'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('23:21:28.512400Z'), TimeResolution.Seconds)
        self.assertEqual(get_time_resolution('06:14:00.000123Z'), TimeResolution.Seconds)

    def test_get_datetime_resolution(self):
        self.assertEqual(get_datetime_resolution('2019-06-05T01:03:11.858714'), TimeResolution.Seconds)
        self.assertEqual(get_datetime_resolution('2019-06-05T01:03:11'), TimeResolution.Seconds)
        self.assertEqual(get_datetime_resolution('2019-06-05T01:03'), TimeResolution.Minutes)
        self.assertEqual(get_datetime_resolution('2019-06-05T01'), TimeResolution.Hours)

    def test_get_time_resolution_badtype(self):
        testtuples = (None, 1, False, 1.234)
        for testtuple in testtuples:
            with self.assertRaises(ValueError):
                get_time_resolution(testtuple)

    def test_get_time_resolution_badstr(self):
        testtuples = ('A6:14:00.000123Z', '06:14:0B', 'bad', '')
        for testtuple in testtuples:
            with self.assertRaises(ISOFormatError):
                get_time_resolution(testtuple)

    def test_get_time_resolution_internal(self):
        self.assertEqual(_get_time_resolution(TimeTuple(hh='01', mm='02', ss='03', tz=None)), TimeResolution.Seconds)
        self.assertEqual(_get_time_resolution(TimeTuple(hh='01', mm='02', ss=None, tz=None)), TimeResolution.Minutes)
        self.assertEqual(_get_time_resolution(TimeTuple(hh='01', mm=None, ss=None, tz=None)), TimeResolution.Hours)