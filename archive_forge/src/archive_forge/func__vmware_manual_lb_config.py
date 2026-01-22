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
def _vmware_manual_lb_config(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareManualLbConfig."""
    kwargs = {'controlPlaneNodePort': flags.Get(args, 'control_plane_node_port'), 'ingressHttpNodePort': flags.Get(args, 'ingress_http_node_port'), 'ingressHttpsNodePort': flags.Get(args, 'ingress_https_node_port'), 'konnectivityServerNodePort': flags.Get(args, 'konnectivity_server_node_port')}
    if flags.IsSet(kwargs):
        return messages.VmwareManualLbConfig(**kwargs)
    return None