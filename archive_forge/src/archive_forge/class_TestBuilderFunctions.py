import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
class TestBuilderFunctions(unittest.TestCase):

    def test_cast(self):
        self.assertEqual(cast('1', int), 1)
        self.assertEqual(cast('-2', int), -2)
        self.assertEqual(cast('3', float), float(3))
        self.assertEqual(cast('-4', float), float(-4))
        self.assertEqual(cast('5.6', float), 5.6)
        self.assertEqual(cast('-7.8', float), -7.8)

    def test_cast_exception(self):
        with self.assertRaises(ISOFormatError):
            cast('asdf', int)
        with self.assertRaises(ISOFormatError):
            cast('asdf', float)

    def test_cast_caughtexception(self):

        def tester(value):
            raise RuntimeError
        with self.assertRaises(ISOFormatError):
            cast('asdf', tester, caughtexceptions=(RuntimeError,))

    def test_cast_thrownexception(self):
        with self.assertRaises(RuntimeError):
            cast('asdf', int, thrownexception=RuntimeError)