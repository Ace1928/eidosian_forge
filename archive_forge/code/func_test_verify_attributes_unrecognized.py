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
def test_verify_attributes_unrecognized(self):
    with testtools.ExpectedException(exc.HTTPBadRequest) as bad_req:
        attributes.AttributeInfo({'attr1': 'foo'}).verify_attributes({'attr1': 'foo', 'attr2': 'bar'})
        self.assertEqual(bad_req.message, "Unrecognized attribute(s) 'attr2'")