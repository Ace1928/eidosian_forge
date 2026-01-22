from neutron_lib.api.definitions import uplink_status_propagation as apidef
from neutron_lib.tests.unit.api.definitions import base
class UplinkStatusPropagationDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = apidef
    extension_attributes = (apidef.PROPAGATE_UPLINK_STATUS,)