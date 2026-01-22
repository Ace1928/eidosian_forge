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
def _AddBGPNodeTaints(bgp_node_pool_config_group):
    """Adds a flag to specify the node taint in the node pool.

  Args:
    bgp_node_pool_config_group: The parent group to add the flags to.
  """
    bgp_node_pool_config_group.add_argument('--bgp-load-balancer-node-taints', metavar='KEY=VALUE:EFFECT', help='Node taint applied to every Kubernetes node in a node pool.', type=arg_parsers.ArgDict())