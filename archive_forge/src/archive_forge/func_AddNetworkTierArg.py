from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddNetworkTierArg(parser):
    parser.add_argument('--network-tier', choices={'PREMIUM': 'High quality, Google-grade network tier.', 'STANDARD': 'Public internet quality.', 'FIXED_STANDARD': 'Public internet quality with fixed bandwidth.'}, type=arg_utils.ChoiceToEnumName, help="\n        Specifies the network tier that will be used to configure the instance\n        network interface. ``NETWORK_TIER'' must be one of: `PREMIUM`,\n        `STANDARD`, `FIXED_STANDARD`. The default value is `PREMIUM`.\n      ")