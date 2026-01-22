import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
class TestTestType(TestCase):
    model = TestType

    def test_provider_tostring(self):
        self.assertEqual(Provider.tostring(TestType.INUSE), 'INUSE')
        self.assertEqual(Provider.tostring(TestType.NOTINUSE), 'NOTINUSE')

    def test_provider_fromstring(self):
        self.assertEqual(TestType.fromstring('inuse'), TestType.INUSE)
        self.assertEqual(TestType.fromstring('NOTINUSE'), TestType.NOTINUSE)

    def test_provider_fromstring_caseinsensitive(self):
        self.assertEqual(TestType.fromstring('INUSE'), TestType.INUSE)
        self.assertEqual(TestType.fromstring('notinuse'), TestType.NOTINUSE)

    def test_compare_as_string(self):
        self.assertTrue(TestType.INUSE == 'inuse')
        self.assertFalse(TestType.INUSE == 'bar')