from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListPool(neutronV20.ListCommand):
    """LBaaS v2 List pools that belong to a given tenant."""
    resource = 'pool'
    shadow_resource = 'lbaas_pool'
    list_columns = ['id', 'name', 'lb_algorithm', 'protocol', 'admin_state_up']
    pagination_support = True
    sorting_support = True