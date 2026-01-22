from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ListMeteringLabel(neutronv20.ListCommand):
    """List metering labels that belong to a given tenant."""
    resource = 'metering_label'
    list_columns = ['id', 'name', 'description', 'shared']
    pagination_support = True
    sorting_support = True