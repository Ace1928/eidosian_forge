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
def test_populate_project_id_not_mandatory(self):
    ctx = context.Context(user_id=None)
    res_dict = {'name': 'test_port'}
    attr_inst = attributes.AttributeInfo({})
    ctx.tenant_id = None
    attr_inst.populate_project_id(ctx, res_dict, True)