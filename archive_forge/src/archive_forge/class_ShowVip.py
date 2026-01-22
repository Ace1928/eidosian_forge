from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class ShowVip(neutronV20.ShowCommand):
    """Show information of a given vip."""
    resource = 'vip'