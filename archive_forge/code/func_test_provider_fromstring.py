import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
def test_provider_fromstring(self):
    self.assertEqual(Provider.fromstring('rackspace'), Provider.RACKSPACE)