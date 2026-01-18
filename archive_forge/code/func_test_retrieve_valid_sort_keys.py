from oslo_utils import uuidutils
import testtools
from webob import exc
from neutron_lib.api import attributes
from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib import constants
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.tests import _base as base
def test_retrieve_valid_sort_keys(self):
    attr_info = {'id': {'visible': True, 'is_sort_key': True}, 'name': {'is_sort_key': True}, 'created_at': {'is_sort_key': False}, 'tenant_id': {'visible': True}}
    expect_val = set(['id', 'name'])
    actual_val = attributes.retrieve_valid_sort_keys(attr_info)
    self.assertEqual(expect_val, actual_val)