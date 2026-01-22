from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddEnableUlaInternalIpv6Arg(parser):
    """Adds the --enable-ula-internal-ipv6 flag."""
    parser.add_argument('--enable-ula-internal-ipv6', action=arg_parsers.StoreTrueFalseAction, help='Enable/disable ULA internal IPv6 on this network. Enabling this\n      feature will assign a /48 from google defined ULA prefix fd20::/20.')