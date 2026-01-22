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
def AddVmwareDataplaneV2Config(parser: parser_arguments.ArgumentInterceptor, for_update=False):
    """Adds flags to specify configurations for Dataplane V2, which is optimized dataplane for Kubernetes networking.

  Args:
    parser: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if for_update:
        return
    vmware_dataplane_v2_config_group = parser.add_group(help='Dataplane V2 configurations')
    _AddEnableDataplaneV2(vmware_dataplane_v2_config_group, for_update=for_update)
    _AddAdvancedNetworking(vmware_dataplane_v2_config_group, for_update=for_update)