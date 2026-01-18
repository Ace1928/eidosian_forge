import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
def test_provider_fromstring_caseinsensitive(self):
    self.assertEqual(TestType.fromstring('INUSE'), TestType.INUSE)
    self.assertEqual(TestType.fromstring('notinuse'), TestType.NOTINUSE)