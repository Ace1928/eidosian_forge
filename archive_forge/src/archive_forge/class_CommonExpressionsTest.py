from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CommonExpressionsTest(ParseTestCase):

    def runTest(self):
        from pyparsing import pyparsing_common
        import ast
        success = pyparsing_common.mac_address.runTests('\n            AA:BB:CC:DD:EE:FF\n            AA.BB.CC.DD.EE.FF\n            AA-BB-CC-DD-EE-FF\n            ')[0]
        self.assertTrue(success, 'error in parsing valid MAC address')
        success = pyparsing_common.mac_address.runTests('\n            # mixed delimiters\n            AA.BB:CC:DD:EE:FF\n            ', failureTests=True)[0]
        self.assertTrue(success, 'error in detecting invalid mac address')
        success = pyparsing_common.ipv4_address.runTests('\n            0.0.0.0\n            1.1.1.1\n            127.0.0.1\n            1.10.100.199\n            255.255.255.255\n            ')[0]
        self.assertTrue(success, 'error in parsing valid IPv4 address')
        success = pyparsing_common.ipv4_address.runTests('\n            # out of range value\n            256.255.255.255\n            ', failureTests=True)[0]
        self.assertTrue(success, 'error in detecting invalid IPv4 address')
        success = pyparsing_common.ipv6_address.runTests('\n            2001:0db8:85a3:0000:0000:8a2e:0370:7334\n            2134::1234:4567:2468:1236:2444:2106\n            0:0:0:0:0:0:A00:1\n            1080::8:800:200C:417A\n            ::A00:1\n\n            # loopback address\n            ::1\n\n            # the null address\n            ::\n\n            # ipv4 compatibility form\n            ::ffff:192.168.0.1\n            ')[0]
        self.assertTrue(success, 'error in parsing valid IPv6 address')
        success = pyparsing_common.ipv6_address.runTests("\n            # too few values\n            1080:0:0:0:8:800:200C\n\n            # too many ::'s, only 1 allowed\n            2134::1234:4567::2444:2106\n            ", failureTests=True)[0]
        self.assertTrue(success, 'error in detecting invalid IPv6 address')
        success = pyparsing_common.number.runTests('\n            100\n            -100\n            +100\n            3.14159\n            6.02e23\n            1e-12\n            ')[0]
        self.assertTrue(success, 'error in parsing valid numerics')
        success = pyparsing_common.sci_real.runTests('\n            1e12\n            -1e12\n            3.14159\n            6.02e23\n            ')[0]
        self.assertTrue(success, 'error in parsing valid scientific notation reals')
        success = pyparsing_common.fnumber.runTests('\n            100\n            -100\n            +100\n            3.14159\n            6.02e23\n            1e-12\n            ')[0]
        self.assertTrue(success, 'error in parsing valid numerics')
        success, results = pyparsing_common.iso8601_date.runTests('\n            1997\n            1997-07\n            1997-07-16\n            ')
        self.assertTrue(success, 'error in parsing valid iso8601_date')
        expected = [('1997', None, None), ('1997', '07', None), ('1997', '07', '16')]
        for r, exp in zip(results, expected):
            self.assertTrue((r[1].year, r[1].month, r[1].day) == exp, 'failed to parse date into fields')
        success, results = pyparsing_common.iso8601_date().addParseAction(pyparsing_common.convertToDate()).runTests('\n            1997-07-16\n            ')
        self.assertTrue(success, 'error in parsing valid iso8601_date with parse action')
        self.assertTrue(results[0][1][0] == datetime.date(1997, 7, 16))
        success, results = pyparsing_common.iso8601_datetime.runTests('\n            1997-07-16T19:20+01:00\n            1997-07-16T19:20:30+01:00\n            1997-07-16T19:20:30.45Z\n            1997-07-16 19:20:30.45\n            ')
        self.assertTrue(success, 'error in parsing valid iso8601_datetime')
        success, results = pyparsing_common.iso8601_datetime().addParseAction(pyparsing_common.convertToDatetime()).runTests('\n            1997-07-16T19:20:30.45\n            ')
        self.assertTrue(success, 'error in parsing valid iso8601_datetime')
        self.assertTrue(results[0][1][0] == datetime.datetime(1997, 7, 16, 19, 20, 30, 450000))
        success = pyparsing_common.uuid.runTests('\n            123e4567-e89b-12d3-a456-426655440000\n            ')[0]
        self.assertTrue(success, 'failed to parse valid uuid')
        success = pyparsing_common.fraction.runTests('\n            1/2\n            -15/16\n            -3/-4\n            ')[0]
        self.assertTrue(success, 'failed to parse valid fraction')
        success = pyparsing_common.mixed_integer.runTests('\n            1/2\n            -15/16\n            -3/-4\n            1 1/2\n            2 -15/16\n            0 -3/-4\n            12\n            ')[0]
        self.assertTrue(success, 'failed to parse valid mixed integer')
        success, results = pyparsing_common.number.runTests('\n            100\n            -3\n            1.732\n            -3.14159\n            6.02e23')
        self.assertTrue(success, 'failed to parse numerics')
        for test, result in results:
            expected = ast.literal_eval(test)
            self.assertEqual(result[0], expected, 'numeric parse failed (wrong value) (%s should be %s)' % (result[0], expected))
            self.assertEqual(type(result[0]), type(expected), 'numeric parse failed (wrong type) (%s should be %s)' % (type(result[0]), type(expected)))