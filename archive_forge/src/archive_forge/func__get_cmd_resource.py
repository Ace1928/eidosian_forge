from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_cmd_resource(obj_type):
    resource = list(RBAC_OBJECTS[obj_type])[0]
    cmd_resource = RBAC_OBJECTS[obj_type][resource]
    return (resource, cmd_resource)