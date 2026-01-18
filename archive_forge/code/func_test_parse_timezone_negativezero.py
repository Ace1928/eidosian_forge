import unittest
import aniso8601
from aniso8601.exceptions import ISOFormatError
from aniso8601.tests.compat import mock
from aniso8601.timezone import parse_timezone
def test_parse_timezone_negativezero(self):
    with self.assertRaises(ISOFormatError):
        parse_timezone('-00:00', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_timezone('-0000', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_timezone('-00', builder=None)