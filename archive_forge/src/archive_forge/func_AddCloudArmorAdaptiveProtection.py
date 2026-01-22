from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddCloudArmorAdaptiveProtection(parser, required=False):
    """Adds the cloud armor adaptive protection arguments to the argparse."""
    parser.add_argument('--enable-layer7-ddos-defense', action='store_true', default=None, required=required, help='Whether to enable Cloud Armor Layer 7 DDoS Defense Adaptive Protection.')
    parser.add_argument('--layer7-ddos-defense-rule-visibility', choices=['STANDARD', 'PREMIUM'], type=lambda x: x.upper(), required=required, metavar='VISIBILITY_TYPE', help='The visibility type indicates whether the rules are opaque or transparent.')