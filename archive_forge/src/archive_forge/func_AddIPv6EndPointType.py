from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddIPv6EndPointType(parser):
    """Adds IPv6 EndPoint flag."""
    choices = ['VM', 'NETLB']
    parser.add_argument('--endpoint-type', choices=choices, type=lambda x: x.upper(), help='        The endpoint type of the external IPv6 address to be reserved.\n      ')