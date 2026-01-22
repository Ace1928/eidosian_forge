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
def _vmware_metal_lb_config_from_flag(self, args: parser_extensions.Namespace) -> messages.VmwareMetalLbConfig:
    vmware_metal_lb_config = messages.VmwareMetalLbConfig()
    for address_pool in args.metal_lb_config_address_pools:
        address_pool_message = messages.VmwareAddressPool(addresses=address_pool.get('addresses', []), avoidBuggyIps=address_pool.get('avoid-buggy-ips', None), manualAssign=address_pool.get('manual-assign', None), pool=address_pool.get('pool', None))
        vmware_metal_lb_config.addressPools.append(address_pool_message)
    return vmware_metal_lb_config