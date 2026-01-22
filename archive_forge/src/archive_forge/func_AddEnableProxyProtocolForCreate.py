from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def AddEnableProxyProtocolForCreate(parser):
    parser.add_argument('--enable-proxy-protocol', action='store_true', default=False, help='      If True, then enable the proxy protocol which is for supplying client\n      TCP/IP address data in TCP connections that traverse proxies on their way\n      to destination servers.\n      ')