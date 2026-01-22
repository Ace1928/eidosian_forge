from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ListVPNService(neutronv20.ListCommand):
    """List VPN service configurations that belong to a given tenant."""
    resource = 'vpnservice'
    list_columns = ['id', 'name', 'router_id', 'status']
    _formatters = {}
    pagination_support = True
    sorting_support = True