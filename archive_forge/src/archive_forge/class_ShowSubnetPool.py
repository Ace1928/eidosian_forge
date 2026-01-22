from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowSubnetPool(neutronV20.ShowCommand):
    """Show information of a given subnetpool."""
    resource = 'subnetpool'