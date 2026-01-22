from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddAddresses(parser):
    """Adds the Addresses flag."""
    parser.add_argument('--addresses', metavar='ADDRESS', type=arg_parsers.ArgList(min_length=1), help="      Ephemeral IP addresses to promote to reserved status. Only addresses\n      that are being used by resources in the project can be promoted. When\n      providing this flag, a parallel list of names for the addresses can\n      be provided. For example,\n\n          $ {command} ADDRESS-1 ADDRESS-2             --addresses 162.222.181.197,162.222.181.198             --region us-central1\n\n      will result in 162.222.181.197 being reserved as\n      'ADDRESS-1' and 162.222.181.198 as 'ADDRESS-2'. If\n      no names are given, server-generated names will be assigned\n      to the IP addresses.\n      ")