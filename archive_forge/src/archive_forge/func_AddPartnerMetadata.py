from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPartnerMetadata(parser, required=True):
    """Adds partner metadata flags to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
    required: A boolean indicates whether the PartnerMetadata is required.
  """
    group = parser.add_group(mutex=False, required=required, help='Partner metadata.')
    group.add_argument('--partner-name', required=required, help='      Plain text name of the Partner providing this attachment. This value\n      may be validated to match approved Partner values.\n      ')
    group.add_argument('--partner-interconnect-name', required=required, help='      Plain text name of the Interconnect this attachment is connected to,\n      as displayed in the Partner\'s portal. For instance "Chicago 1".\n      ')
    group.add_argument('--partner-portal-url', required=required, help="      URL of the Partner's portal for this Attachment. The Partner may wish\n      to customize this to be a deep-link to the specific resource on the\n      Partner portal. This value may be validated to match approved Partner\n      values.\n      ")