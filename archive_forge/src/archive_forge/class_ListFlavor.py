import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListFlavor(neutronV20.ListCommand):
    """List Neutron service flavors."""
    resource = 'flavor'
    list_columns = ['id', 'name', 'service_type', 'enabled']
    pagination_support = True
    sorting_support = True