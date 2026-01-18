import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
def test_provider_tostring(self):
    self.assertEqual(Provider.tostring(Provider.RACKSPACE), 'RACKSPACE')