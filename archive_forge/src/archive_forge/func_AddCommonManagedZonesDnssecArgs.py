from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def AddCommonManagedZonesDnssecArgs(parser, messages):
    """Add Common DNSSEC flags for the managed-zones group."""
    GetDnsSecStateFlagMapper(messages).choice_arg.AddToParser(parser)
    GetDoeFlagMapper(messages).choice_arg.AddToParser(parser)
    GetKeyAlgorithmFlag('ksk', messages).choice_arg.AddToParser(parser)
    GetKeyAlgorithmFlag('zsk', messages).choice_arg.AddToParser(parser)
    parser.add_argument('--ksk-key-length', type=int, help='Length of the key-signing key in bits. Requires DNSSEC enabled.')
    parser.add_argument('--zsk-key-length', type=int, help='Length of the zone-signing key in bits. Requires DNSSEC enabled.')