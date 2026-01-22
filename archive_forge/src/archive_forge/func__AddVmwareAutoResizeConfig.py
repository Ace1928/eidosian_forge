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
def _AddVmwareAutoResizeConfig(vmware_control_plane_node_config_group, for_update=False):
    """Adds flags to specify control plane auto resizing configurations.

  Args:
    vmware_control_plane_node_config_group: The parent group to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    vmware_auto_resize_config_group = vmware_control_plane_node_config_group.add_group(help='Auto resize configurations')
    if for_update:
        enable_auto_resize_mutex_group = vmware_auto_resize_config_group.add_group(mutex=True)
        surface = enable_auto_resize_mutex_group
    else:
        surface = vmware_auto_resize_config_group
    surface.add_argument('--enable-auto-resize', action='store_true', help='Enable controle plane node auto resize.')
    if for_update:
        surface.add_argument('--disable-auto-resize', action='store_true', help='Disable controle plane node auto resize.')