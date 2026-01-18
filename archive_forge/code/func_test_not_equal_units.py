from unittest import mock
import ddt
from cinderclient.tests.unit import utils
from cinderclient.v3 import limits
def test_not_equal_units(self):
    l1 = _get_default_RateLimit()
    l2 = _get_default_RateLimit(unit='unit2')
    self.assertNotEqual(l1, l2)