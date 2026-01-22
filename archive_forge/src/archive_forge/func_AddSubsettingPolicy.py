from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSubsettingPolicy(parser):
    parser.add_argument('--subsetting-policy', choices=['NONE', 'CONSISTENT_HASH_SUBSETTING'], type=lambda x: x.replace('-', '_').upper(), default='NONE', help='      Specifies the algorithm used for subsetting.\n      Default value is NONE which implies that subsetting is disabled.\n      For Layer 4 Internal Load Balancing, if subsetting is enabled,\n      only the algorithm CONSISTENT_HASH_SUBSETTING can be specified.\n      ')