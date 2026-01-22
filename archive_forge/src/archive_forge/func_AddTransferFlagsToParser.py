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
def AddTransferFlagsToParser(parser):
    """Get flags for transferring a domain.

  Args:
    parser: argparse parser to which to add these flags.
  """
    _AddDNSSettingsFlagsToParser(parser, mutation_op=MutationOp.TRANSFER)
    _AddContactSettingsFlagsToParser(parser, mutation_op=MutationOp.TRANSFER)
    _AddPriceFlagsToParser(parser, MutationOp.TRANSFER)
    help_text = "    A file containing the authorizaton code. In most cases, you must provide an\n    authorization code from the domain's current registrar to transfer the\n    domain.\n\n    Examples of file contents:\n\n    ```\n    5YcCd!X&W@q0Xozj\n    ```\n    "
    base.Argument('--authorization-code-from-file', help=help_text, metavar='AUTHORIZATION_CODE_FILE_NAME', category=base.COMMONLY_USED_FLAGS).AddToParser(parser)
    messages = apis.GetMessagesModule('domains', API_VERSION_FOR_FLAGS)
    notice_choices = ContactNoticeEnumMapper(messages).choices.copy()
    base.Argument('--notices', help='Notices about special properties of certain domains or contacts.', metavar='NOTICE', type=arg_parsers.ArgList(element_type=str, choices=notice_choices)).AddToParser(parser)