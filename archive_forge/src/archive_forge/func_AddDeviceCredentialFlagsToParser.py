from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from six.moves import map  # pylint: disable=redefined-builtin
from the device. This flag can be specified multiple times to add multiple
def AddDeviceCredentialFlagsToParser(parser, combine_flags=True, only_modifiable=False):
    """Get credentials-related flags.

  Adds one of the following:

    * --public-key path=PATH,type=TYPE,expiration-time=EXPIRATION_TIME
    * --path=PATH --type=TYPE --expiration-time=EXPIRATION_TIME

  depending on the value of combine_flags.

  Args:
    parser: argparse parser to which to add these flags.
    combine_flags: bool, whether to combine these flags into one --public-key
      flag or to leave them separate.
    only_modifiable: bool, whether to include all flags or just those that can
      be modified after creation.
  """
    for f in _GetDeviceCredentialFlags(combine_flags, only_modifiable):
        f.AddToParser(parser)