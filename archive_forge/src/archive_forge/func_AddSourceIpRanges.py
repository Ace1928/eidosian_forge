from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def AddSourceIpRanges(parser):
    """Adds source-ip-ranges flag to the argparse.

  Args:
    parser: The parser that parses args from user input.
  """
    parser.add_argument('--source-ip-ranges', metavar='SOURCE_IP_RANGE,[...]', type=SourceIpRangesParser, default=None, help="      List of comma-separated IP addresses or IP ranges. If set, this forwarding\n      rule only forwards traffic when the packet's source IP address matches one\n      of the IP ranges set here.\n      ")