from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddValidateOnly(parser, help_action):
    """Add the --validate-only argument.

  Args:
    parser: The argparse.parser to add the argument to.
    help_action: str, describes the action that will be validated.
  """
    parser.add_argument('--validate-only', action='store_true', help="Validate the {}, but don't actually perform it.".format(help_action))