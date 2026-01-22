from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
class ListPeers(neutronv20.ListCommand):
    """List BGP peers."""
    resource = 'bgp_peer'
    list_columns = ['id', 'name', 'peer_ip', 'remote_as']
    pagination_support = True
    sorting_support = True