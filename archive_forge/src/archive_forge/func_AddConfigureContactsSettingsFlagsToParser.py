from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddConfigureContactsSettingsFlagsToParser(parser):
    """Get flags for changing contact settings.

  Args:
    parser: argparse parser to which to add these flags.
  """
    _AddContactSettingsFlagsToParser(parser, mutation_op=MutationOp.UPDATE)
    messages = apis.GetMessagesModule('domains', API_VERSION_FOR_FLAGS)
    base.Argument('--notices', help='Notices about special properties of contacts.', metavar='NOTICE', type=arg_parsers.ArgList(element_type=str, choices=ContactNoticeEnumMapper(messages).choices)).AddToParser(parser)