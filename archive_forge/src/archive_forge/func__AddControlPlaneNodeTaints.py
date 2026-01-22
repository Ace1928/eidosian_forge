from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddControlPlaneNodeTaints(bare_metal_node_config_group):
    """Adds a flag to specify the node taint in the node pool.

  Args:
    bare_metal_node_config_group: The parent group to add the flags to.
  """
    bare_metal_node_config_group.add_argument('--control-plane-node-taints', metavar='KEY=VALUE:EFFECT', help='Node taint applied to every Kubernetes node in a node pool.', type=arg_parsers.ArgDict())