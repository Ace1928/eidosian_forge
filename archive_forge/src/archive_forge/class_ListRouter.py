import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class ListRouter(neutronV20.ListCommand):
    """List routers that belong to a given tenant."""
    resource = 'router'
    _formatters = {'external_gateway_info': _format_external_gateway_info}
    list_columns = ['id', 'name', 'external_gateway_info', 'distributed', 'ha']
    pagination_support = True
    sorting_support = True