import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListFlavorProfile(neutronV20.ListCommand):
    """List Neutron service flavor profiles."""
    resource = 'service_profile'
    list_columns = ['id', 'description', 'enabled', 'metainfo']
    pagination_support = True
    sorting_support = True