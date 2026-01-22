from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator, Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.api_lib.container.vmware import version_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _vmware_address_pool(self, address_pool) -> messages.VmwareAddressPool:
    """Constructs proto message VmwareAddressPool."""
    addresses = address_pool.get('addresses', [])
    if not addresses:
        raise InvalidConfigFile('Missing field [addresses] in Metal LB configuration file.')
    avoid_buggy_ips = address_pool.get('avoidBuggyIPs', None)
    manual_assign = address_pool.get('manualAssign', None)
    pool = address_pool.get('pool', None)
    if not pool:
        raise InvalidConfigFile('Missing field [pool] in Metal LB configuration file.')
    kwargs = {'addresses': addresses, 'avoidBuggyIps': avoid_buggy_ips, 'manualAssign': manual_assign, 'pool': pool}
    return messages.VmwareAddressPool(**kwargs)