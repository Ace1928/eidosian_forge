from neutron_lib.api.definitions import extraroute
from neutron_lib.tests.unit.api.definitions import base
class ExtrarouteDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = extraroute
    extension_attributes = (extraroute.ROUTES,)