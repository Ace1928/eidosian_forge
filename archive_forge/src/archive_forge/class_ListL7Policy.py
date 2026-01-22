from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListL7Policy(neutronV20.ListCommand):
    """LBaaS v2 List L7 policies that belong to a given listener."""
    resource = 'l7policy'
    shadow_resource = 'lbaas_l7policy'
    pagination_support = True
    sorting_support = True
    list_columns = ['id', 'name', 'action', 'redirect_pool_id', 'redirect_url', 'position', 'admin_state_up', 'status']