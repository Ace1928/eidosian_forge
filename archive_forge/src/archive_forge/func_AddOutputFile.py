from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddOutputFile(parser, help_action):
    """Add an output file argument.

  Args:
    parser: The argparse.parser to add the output file argument to.
    help_action: str, describes the action of what will be stored.
  """
    parser.add_argument('--output-file', help='Path to the output file {}.'.format(help_action))