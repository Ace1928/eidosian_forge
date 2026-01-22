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
def AddVmwareControlPlaneNodeConfig(parser: parser_arguments.ArgumentInterceptor, for_update=False, release_track: base.ReleaseTrack=None):
    """Adds flags to specify VMware user cluster control plane node configurations.

  Args:
    parser: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
    release_track: base.ReleaseTrack, determine the flag scope based on release
      tracks.
  """
    vmware_control_plane_node_config_group = parser.add_group(help='Control plane node configurations')
    vmware_control_plane_node_config_group.add_argument('--cpus', type=int, help='Number of CPUs for each admin cluster node that serve as control planes for this VMware user cluster. (default: 4 CPUs)')
    vmware_control_plane_node_config_group.add_argument('--memory', type=arg_parsers.BinarySize(default_unit='MB', type_abbr='MB'), help='Megabytes of memory for each admin cluster node that serves as a control plane for this VMware User Cluster (default: 8192 MB memory).')
    if not for_update:
        vmware_control_plane_node_config_group.add_argument('--replicas', type=int, help='Number of control plane nodes for this VMware user cluster. (default: 1 replica).')
    _AddVmwareAutoResizeConfig(vmware_control_plane_node_config_group, for_update=for_update)
    _AddVmwareControlPlaneVsphereConfig(vmware_control_plane_node_config_group, release_track=release_track)