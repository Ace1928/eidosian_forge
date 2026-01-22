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
def AddAssignIp(parser, hidden=False):
    parser.add_argument('--assign-ip', help='Assign a public IP address to the instance. This is a public, externally available IPv4 address that you can use to connect to your instance when properly authorized.', hidden=hidden, action=arg_parsers.StoreTrueFalseAction)