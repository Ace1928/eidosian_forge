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
def test_populate_project_id_admin_req(self):
    tenant_id_1 = uuidutils.generate_uuid()
    tenant_id_2 = uuidutils.generate_uuid()
    ctx = context.Context(user_id=None, tenant_id=tenant_id_1)
    res_dict = {'tenant_id': tenant_id_2}
    attr_inst = attributes.AttributeInfo({})
    self.assertRaises(exc.HTTPBadRequest, attr_inst.populate_project_id, ctx, res_dict, None)
    ctx.is_admin = True
    attr_inst.populate_project_id(ctx, res_dict, is_create=False)