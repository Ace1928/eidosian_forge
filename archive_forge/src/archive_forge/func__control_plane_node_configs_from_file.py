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
def _control_plane_node_configs_from_file(self, args: parser_extensions.Namespace):
    """Constructs proto message field node_configs."""
    if not args.control_plane_node_configs_from_file:
        return []
    control_plane_node_configs = args.control_plane_node_configs_from_file.get('nodeConfigs', [])
    if not control_plane_node_configs:
        raise exceptions.BadArgumentException('--control_plane_node_configs_from_file', 'Missing field [nodeConfigs] in Control Plane Node configs file.')
    control_plane_node_configs_messages = []
    for control_plane_node_config in control_plane_node_configs:
        control_plane_node_configs_messages.append(self._control_plane_node_config(control_plane_node_config))
    return control_plane_node_configs_messages