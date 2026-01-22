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
def AddVmwareAutoRepairConfig(parser: parser_arguments.ArgumentInterceptor, for_update=False):
    """Adds flags to specify auto-repair configurations.

  Args:
    parser: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    vmware_auto_repair_config_group = parser.add_group(help='Auto-repair configurations')
    if for_update:
        enable_auto_repair_mutex_group = vmware_auto_repair_config_group.add_group(mutex=True)
        surface = enable_auto_repair_mutex_group
    else:
        surface = vmware_auto_repair_config_group
    surface.add_argument('--enable-auto-repair', action='store_true', help='If set, deploy the cluster-health-controller.')
    if for_update:
        surface.add_argument('--disable-auto-repair', action='store_true', help='If set, disables auto repair.')