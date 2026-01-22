from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddPodAddressCIDRBlocks(bare_metal_island_mode_cidr_config_group):
    """Adds a flag to specify the IPv4 address ranges used in the pods in the cluster.

  Args:
    bare_metal_island_mode_cidr_config_group: The parent group to add the flag
      to.
  """
    bare_metal_island_mode_cidr_config_group.add_argument('--island-mode-pod-address-cidr-blocks', metavar='POD_ADDRESS', required=True, type=arg_parsers.ArgList(min_length=1), help='IPv4 address range for all pods in the cluster.')