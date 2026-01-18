from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_set_with_invalid_keys(self):
    invalid_keys = ['/1', '?1', '%1', '<', '>']
    f = self.cs.flavors.get(1)
    for key in invalid_keys:
        self.assertRaises(exceptions.CommandError, f.set_keys, {key: 'v1'})