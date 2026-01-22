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
def AddVmwareNetworkConfig(parser: parser_arguments.ArgumentInterceptor, for_update=False):
    """Adds network config related flags.

  Args:
    parser: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    required = False if for_update else True
    vmware_network_config_group = parser.add_group(help='VMware User Cluster network configurations', required=required)
    _AddServiceAddressCidrBlocks(vmware_network_config_group, for_update=for_update)
    _AddPodAddressCidrBlocks(vmware_network_config_group, for_update=for_update)
    _AddIpConfiguration(vmware_network_config_group, for_update=for_update)
    _AddVmwareHostConfig(vmware_network_config_group, for_update=for_update)
    if not for_update:
        _AddVmwareControlPlaneV2Config(vmware_network_config_group)