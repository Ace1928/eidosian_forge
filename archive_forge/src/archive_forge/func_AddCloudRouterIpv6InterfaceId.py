from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCloudRouterIpv6InterfaceId(parser):
    """Adds cloud router ipv6 interface id flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--cloud-router-ipv6-interface-id', metavar='INTERFACE_ID', help='The `cloud-router-ipv6-interface-id` field is not available.')