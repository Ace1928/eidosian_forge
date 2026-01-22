from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowHealthMonitor(neutronV20.ShowCommand):
    """LBaaS v2 Show information of a given healthmonitor."""
    resource = 'healthmonitor'
    shadow_resource = 'lbaas_healthmonitor'