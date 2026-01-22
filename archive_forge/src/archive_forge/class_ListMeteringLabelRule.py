from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ListMeteringLabelRule(neutronv20.ListCommand):
    """List metering labels that belong to a given label."""
    resource = 'metering_label_rule'
    list_columns = ['id', 'excluded', 'direction', 'remote_ip_prefix']
    pagination_support = True
    sorting_support = True