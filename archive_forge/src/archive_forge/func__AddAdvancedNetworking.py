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
def _AddAdvancedNetworking(vmware_dataplane_v2_config_group, for_update=False):
    """Adds flags to specify advanced networking.

  Args:
    vmware_dataplane_v2_config_group: The parent group to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if for_update:
        return
    vmware_dataplane_v2_config_group.add_argument('--enable-advanced-networking', action='store_true', help='If set, enable advanced networking. Requires dataplane_v2_enabled to be set true.')