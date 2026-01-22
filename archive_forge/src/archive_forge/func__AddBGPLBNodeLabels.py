from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddBGPLBNodeLabels(bare_metal_bgp_lb_node_config):
    """Adds a flag to assign labels to nodes in a BGP LB node pool.

  Args:
    bare_metal_bgp_lb_node_config: The parent group to add the flags to.
  """
    bare_metal_bgp_lb_node_config.add_argument('--bgp-lb-load-balancer-node-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='Labels assigned to nodes of a BGP LB node pool.')