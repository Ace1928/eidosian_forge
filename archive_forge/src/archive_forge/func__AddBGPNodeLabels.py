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
def _AddBGPNodeLabels(bgp_node_pool_config_group):
    """Adds a flag to assign labels to nodes in a BGP node pool.

  Args:
    bgp_node_pool_config_group: The parent group to add the flags to.
  """
    bgp_node_pool_config_group.add_argument('--bgp-load-balancer-node-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='Labels assigned to nodes of a BGP node pool.')