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
def _vmware_control_plane_vsphere_config(self, args: parser_extensions.Namespace) -> Optional[messages.VmwareControlPlaneVsphereConfig]:
    """Constructs proto message VmwareControlPlaneVsphereConfig."""
    if 'control_plane_vsphere_config' not in args.GetSpecifiedArgsDict():
        return None
    kwargs = {'datastore': args.control_plane_vsphere_config.get('datastore', None), 'storagePolicyName': args.control_plane_vsphere_config.get('storage-policy-name', None)}
    return messages.VmwareControlPlaneVsphereConfig(**kwargs)