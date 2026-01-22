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
def _vmware_network_config(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareNetworkConfig."""
    kwargs = {'serviceAddressCidrBlocks': flags.Get(args, 'service_address_cidr_blocks', []), 'podAddressCidrBlocks': flags.Get(args, 'pod_address_cidr_blocks', []), 'staticIpConfig': self._vmware_static_ip_config(args), 'dhcpIpConfig': self._vmware_dhcp_ip_config(args), 'hostConfig': self._vmware_host_config(args), 'controlPlaneV2Config': self._vmware_control_plane_v2_config(args)}
    if any(kwargs.values()):
        return messages.VmwareNetworkConfig(**kwargs)
    return None