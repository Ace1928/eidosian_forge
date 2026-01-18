import copy
from neutron_lib import constants
from neutron_lib.tests import _base as base
def test_sentinel_constant(self):
    foo = constants.Sentinel()
    bar = copy.deepcopy(foo)
    self.assertEqual(id(foo), id(bar))