from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteSubnetPool(neutronV20.DeleteCommand):
    """Delete a given subnetpool."""
    resource = 'subnetpool'