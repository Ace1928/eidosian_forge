from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteRBACPolicy(neutronV20.DeleteCommand):
    """Delete a RBAC policy."""
    resource = 'rbac_policy'
    allow_names = False