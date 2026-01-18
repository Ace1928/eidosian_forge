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
def test_populate_project_info_ids_match(self):
    project_id = uuidutils.generate_uuid()
    attrs_in = {'project_id': project_id, 'tenant_id': project_id}
    attrs_out = attributes.populate_project_info(attrs_in)
    self.assertEqual(attrs_in, attrs_out)