from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class DeleteEndpointGroup(neutronv20.DeleteCommand):
    """Delete a given VPN endpoint group."""
    resource = 'endpoint_group'