from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteListener(neutronV20.DeleteCommand):
    """LBaaS v2 Delete a given listener."""
    resource = 'listener'