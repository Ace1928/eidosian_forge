import unittest
from aniso8601.decimalfraction import normalize
class TestDecimalFractionFunctions(unittest.TestCase):

    def test_normalize(self):
        self.assertEqual(normalize(''), '')
        self.assertEqual(normalize('12.34'), '12.34')
        self.assertEqual(normalize('123,45'), '123.45')
        self.assertEqual(normalize('123,45,67'), '123.45.67')