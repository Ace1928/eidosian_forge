from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddEnableMulticast(parser, update=False):
    """Adds enableMulticast flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
    update: A boolean indicates whether the incoming request is an update.
  """
    if update:
        help_text = '      When enabled, the attachment will be able to carry multicast traffic.\n      Use --no-enable-multicast to disable it.\n      '
    else:
        help_text = '      If supplied, the attachment will be able to carry multicast traffic.\n      If not provided on creation, defaults to disabled. Use\n      --no-enable-multicast to disable it.\n      '
    parser.add_argument('--enable-multicast', default=None, action='store_true', help=help_text)