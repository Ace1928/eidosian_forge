from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddVmwareControlPlaneV2Config(vmware_network_config_group, for_update=False):
    """Adds a flag for control_plane_v2_config message.

  Args:
    vmware_network_config_group: The parent group to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if for_update:
        return
    vmware_control_plane_v2_config_group = vmware_network_config_group.add_group(help='Control plane v2 mode configurations.')
    help_text = "\nStatic IP addresses for the control plane nodes. The number of IP addresses should match the number of replicas for the control plane nodes, specified by `--replicas`.\n\nTo specify the control plane IP block,\n\n```\n$ gcloud {command}\n    --control-plane-ip-block 'gateway=192.168.0.1,netmask=255.255.255.0,ips=192.168.1.1;0.0.0.0 localhost;'\n```\n\n  "
    vmware_control_plane_v2_config_group.add_argument('--control-plane-ip-block', help=help_text, type=arg_parsers.ArgDict(spec={'gateway': str, 'netmask': str, 'ips': arg_parsers.ArgList(element_type=_ParseControlPlaneIpBlock, custom_delim_char=';')}))