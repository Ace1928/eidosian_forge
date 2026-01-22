from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListSubnetPool(neutronV20.ListCommand):
    """List subnetpools that belong to a given tenant."""
    _formatters = {'prefixes': _format_prefixes}
    resource = 'subnetpool'
    list_columns = ['id', 'name', 'prefixes', 'default_prefixlen', 'address_scope_id', 'is_default']
    pagination_support = True
    sorting_support = True