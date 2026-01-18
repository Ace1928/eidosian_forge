from unittest import mock
import ddt
from cinderclient.tests.unit import utils
from cinderclient.v3 import limits
def test_not_equal_names(self):
    l1 = limits.AbsoluteLimit('name1', 'value1')
    l2 = limits.AbsoluteLimit('name2', 'value1')
    self.assertNotEqual(l1, l2)