from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowListener(neutronV20.ShowCommand):
    """LBaaS v2 Show information of a given listener."""
    resource = 'listener'