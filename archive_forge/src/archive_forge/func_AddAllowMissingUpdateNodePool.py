from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAllowMissingUpdateNodePool(parser: parser_arguments.ArgumentInterceptor):
    """Adds a flag to enable allow missing in an update node pool request.

  If set to true, and the Bare Metal Node Pool is not found, the request will
  create a new Bare Metal Node Pool with the provided configuration. The user
  must have both create and update permission to call Update with allow_missing
  set to true.

  Args:
    parser: The argparse parser to add the flag to.
  """
    parser.add_argument('--allow-missing', action='store_true', help='If set, and the Anthos cluster on bare metal is not found, the update request will try to create a new cluster with the provided configuration.')