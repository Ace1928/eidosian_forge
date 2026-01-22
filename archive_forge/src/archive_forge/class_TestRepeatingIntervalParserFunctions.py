import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
class TestRepeatingIntervalParserFunctions(unittest.TestCase):

    def test_get_interval_resolution_date(self):
        self.assertEqual(get_repeating_interval_resolution('R/P1.5Y/2018'), IntervalResolution.Year)
        self.assertEqual(get_repeating_interval_resolution('R1/P1.5Y/2018-03'), IntervalResolution.Month)
        self.assertEqual(get_repeating_interval_resolution('R2/P1.5Y/2018-03-06'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R3/P1.5Y/2018W01'), IntervalResolution.Week)
        self.assertEqual(get_repeating_interval_resolution('R4/P1.5Y/2018-306'), IntervalResolution.Ordinal)
        self.assertEqual(get_repeating_interval_resolution('R5/P1.5Y/2018W012'), IntervalResolution.Weekday)
        self.assertEqual(get_repeating_interval_resolution('R/2018/P1.5Y'), IntervalResolution.Year)
        self.assertEqual(get_repeating_interval_resolution('R1/2018-03/P1.5Y'), IntervalResolution.Month)
        self.assertEqual(get_repeating_interval_resolution('R2/2018-03-06/P1.5Y'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R3/2018W01/P1.5Y'), IntervalResolution.Week)
        self.assertEqual(get_repeating_interval_resolution('R4/2018-306/P1.5Y'), IntervalResolution.Ordinal)
        self.assertEqual(get_repeating_interval_resolution('R5/2018W012/P1.5Y'), IntervalResolution.Weekday)

    def test_get_interval_resolution_time(self):
        self.assertEqual(get_repeating_interval_resolution('R/P1M/1981-04-05T01'), IntervalResolution.Hours)
        self.assertEqual(get_repeating_interval_resolution('R1/P1M/1981-04-05T01:01'), IntervalResolution.Minutes)
        self.assertEqual(get_repeating_interval_resolution('R2/P1M/1981-04-05T01:01:00'), IntervalResolution.Seconds)
        self.assertEqual(get_repeating_interval_resolution('R/1981-04-05T01/P1M'), IntervalResolution.Hours)
        self.assertEqual(get_repeating_interval_resolution('R1/1981-04-05T01:01/P1M'), IntervalResolution.Minutes)
        self.assertEqual(get_repeating_interval_resolution('R2/1981-04-05T01:01:00/P1M'), IntervalResolution.Seconds)

    def test_get_interval_resolution_duration(self):
        self.assertEqual(get_repeating_interval_resolution('R/2014-11-12/P1Y2M3D'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R1/2014-11-12/P1Y2M'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R2/2014-11-12/P1Y'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R3/2014-11-12/P1W'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R4/2014-11-12/P1Y2M3DT4H'), IntervalResolution.Hours)
        self.assertEqual(get_repeating_interval_resolution('R5/2014-11-12/P1Y2M3DT4H54M'), IntervalResolution.Minutes)
        self.assertEqual(get_repeating_interval_resolution('R6/2014-11-12/P1Y2M3DT4H54M6S'), IntervalResolution.Seconds)
        self.assertEqual(get_repeating_interval_resolution('R/P1Y2M3D/2014-11-12'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R1/P1Y2M/2014-11-12'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R2/P1Y/2014-11-12'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R3/P1W/2014-11-12'), IntervalResolution.Day)
        self.assertEqual(get_repeating_interval_resolution('R4/P1Y2M3DT4H/2014-11-12'), IntervalResolution.Hours)
        self.assertEqual(get_repeating_interval_resolution('R5/P1Y2M3DT4H54M/2014-11-12'), IntervalResolution.Minutes)
        self.assertEqual(get_repeating_interval_resolution('R6/P1Y2M3DT4H54M6S/2014-11-12'), IntervalResolution.Seconds)

    def test_parse_repeating_interval(self):
        with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
            expectedargs = {'R': False, 'Rnn': '3', 'interval': IntervalTuple(DateTuple('1981', '04', '05', None, None, None), None, DurationTuple(None, None, None, '1', None, None, None))}
            mockBuilder.return_value = expectedargs
            result = parse_repeating_interval('R3/1981-04-05/P1D')
            self.assertEqual(result, expectedargs)
            mockBuilder.assert_called_once_with(**expectedargs)
        with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
            expectedargs = {'R': False, 'Rnn': '11', 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
            mockBuilder.return_value = expectedargs
            result = parse_repeating_interval('R11/PT1H2M/1980-03-05T01:01:00')
            self.assertEqual(result, expectedargs)
            mockBuilder.assert_called_once_with(**expectedargs)
        with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
            expectedargs = {'R': False, 'Rnn': '2', 'interval': IntervalTuple(DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DatetimeTuple(DateTuple('1981', '04', '05', None, None, None), TimeTuple('01', '01', '00', None)), None)}
            mockBuilder.return_value = expectedargs
            result = parse_repeating_interval('R2--1980-03-05T01:01:00--1981-04-05T01:01:00', intervaldelimiter='--')
            self.assertEqual(result, expectedargs)
            mockBuilder.assert_called_once_with(**expectedargs)
        with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
            expectedargs = {'R': False, 'Rnn': '2', 'interval': IntervalTuple(DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DatetimeTuple(DateTuple('1981', '04', '05', None, None, None), TimeTuple('01', '01', '00', None)), None)}
            mockBuilder.return_value = expectedargs
            result = parse_repeating_interval('R2/1980-03-05 01:01:00/1981-04-05 01:01:00', datetimedelimiter=' ')
            self.assertEqual(result, expectedargs)
            mockBuilder.assert_called_once_with(**expectedargs)
        with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
            expectedargs = {'R': True, 'Rnn': None, 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
            mockBuilder.return_value = expectedargs
            result = parse_repeating_interval('R/PT1H2M/1980-03-05T01:01:00')
            self.assertEqual(result, expectedargs)
            mockBuilder.assert_called_once_with(**expectedargs)

    def test_parse_repeating_interval_mockbuilder(self):
        mockBuilder = mock.Mock()
        args = {'R': False, 'Rnn': '3', 'interval': IntervalTuple(DateTuple('1981', '04', '05', None, None, None), None, DurationTuple(None, None, None, '1', None, None, None))}
        mockBuilder.build_repeating_interval.return_value = args
        result = parse_repeating_interval('R3/1981-04-05/P1D', builder=mockBuilder)
        self.assertEqual(result, args)
        mockBuilder.build_repeating_interval.assert_called_once_with(**args)
        mockBuilder = mock.Mock()
        args = {'R': False, 'Rnn': '11', 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
        mockBuilder.build_repeating_interval.return_value = args
        result = parse_repeating_interval('R11/PT1H2M/1980-03-05T01:01:00', builder=mockBuilder)
        self.assertEqual(result, args)
        mockBuilder.build_repeating_interval.assert_called_once_with(**args)
        mockBuilder = mock.Mock()
        args = {'R': True, 'Rnn': None, 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
        mockBuilder.build_repeating_interval.return_value = args
        result = parse_repeating_interval('R/PT1H2M/1980-03-05T01:01:00', builder=mockBuilder)
        self.assertEqual(result, args)
        mockBuilder.build_repeating_interval.assert_called_once_with(**args)

    def test_parse_repeating_interval_badtype(self):
        testtuples = (None, 1, False, 1.234)
        for testtuple in testtuples:
            with self.assertRaises(ValueError):
                parse_repeating_interval(testtuple, builder=None)

    def test_parse_repeating_interval_baddelimiter(self):
        testtuples = ('R,PT1H2M,1980-03-05T01:01:00', 'R3 1981-04-05 P1D')
        for testtuple in testtuples:
            with self.assertRaises(ISOFormatError):
                parse_repeating_interval(testtuple, builder=None)

    def test_parse_repeating_interval_suffixgarbage(self):
        with self.assertRaises(ISOFormatError):
            parse_repeating_interval('R3/1981-04-05/P1Dasdf', builder=None)
        with self.assertRaises(ISOFormatError):
            parse_repeating_interval('R3/1981-04-05/P0003-06-04T12:30:05.5asdfasdf', builder=None)

    def test_parse_repeating_interval_badstr(self):
        testtuples = ('bad', '')
        for testtuple in testtuples:
            with self.assertRaises(ISOFormatError):
                parse_repeating_interval(testtuple, builder=None)