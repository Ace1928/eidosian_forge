from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAllowMissingNodePoolFlag(parser: parser_arguments.ArgumentInterceptor) -> None:
    """Adds a flag for the node pool operation to return success and perform no action when there is no matching node pool.

  Args:
    parser: The argparse parser to add the flag to.
  """
    parser.add_argument('--allow-missing', action='store_true', help='If set, and the Bare Metal Standalone Node Pool is not found, the request will succeed but no action will be taken.')