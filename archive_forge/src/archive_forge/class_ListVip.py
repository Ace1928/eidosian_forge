from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class ListVip(neutronV20.ListCommand):
    """List vips that belong to a given tenant."""
    resource = 'vip'
    list_columns = ['id', 'name', 'algorithm', 'address', 'protocol', 'admin_state_up', 'status']
    pagination_support = True
    sorting_support = True