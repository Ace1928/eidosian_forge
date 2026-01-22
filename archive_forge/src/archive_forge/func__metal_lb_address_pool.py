from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _metal_lb_address_pool(self, address_pool):
    """Constructs proto message BareMetalStandaloneLoadBalancerAddressPool."""
    addresses = address_pool.get('addresses', [])
    if not addresses:
        self._raise_bad_argument_exception_error('--metal_lb_address_pools_from_file', 'addresses', 'Metal LB address pools file')
    pool = address_pool.get('pool', None)
    if not pool:
        self._raise_bad_argument_exception_error('--metal_lb_address_pools_from_file', 'pool', 'Metal LB address pools file')
    kwargs = {'addresses': addresses, 'avoidBuggyIps': address_pool.get('avoidBuggyIPs', None), 'manualAssign': address_pool.get('manualAssign', None), 'pool': pool}
    return messages.BareMetalStandaloneLoadBalancerAddressPool(**kwargs)