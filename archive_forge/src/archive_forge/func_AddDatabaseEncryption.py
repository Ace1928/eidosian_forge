from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddDatabaseEncryption(parser):
    """Adds database encryption flags.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    parser.add_argument('--database-encryption-key-id', help='URL the of the Azure Key Vault key (with its version) to use to encrypt / decrypt cluster secrets.')