from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddActiveDirectoryDomain(parser, hidden=False):
    """Adds the '--active-directory-domain' flag to the parser.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    help_text = 'Managed Service for Microsoft Active Directory domain this instance is joined to. Only available for SQL Server instances.'
    parser.add_argument('--active-directory-domain', help=help_text, hidden=hidden)