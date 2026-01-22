from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def AddIPProtocols(parser, support_all_protocol, support_l3_default):
    """Adds IP protocols flag, with values available in the given version.

  Args:
    parser: The parser that parses args from user input.
    support_all_protocol: Whether to include "ALL" in the protocols list.
    support_l3_default: Whether to include "L3_DEFAULT" in the protocols list.
  """
    protocols = ['AH', 'ESP', 'ICMP', 'SCTP', 'TCP', 'UDP']
    if support_l3_default:
        protocols.append('L3_DEFAULT')
    if support_all_protocol:
        protocols.append('ALL')
        if support_l3_default:
            help_str = '        IP protocol that the rule will serve. The default is `TCP`.\n\n        Note that if the load-balancing scheme is `INTERNAL`, the protocol must\n        be one of: `TCP`, `UDP`, `ALL`, `L3_DEFAULT`.\n\n        For a load-balancing scheme that is `EXTERNAL`, all IP_PROTOCOL\n        options other than `ALL` are valid.\n        '
        else:
            help_str = '        IP protocol that the rule will serve. The default is `TCP`.\n\n        Note that if the load-balancing scheme is `INTERNAL`, the protocol must\n        be one of: `TCP`, `UDP`, `ALL`.\n\n        For a load-balancing scheme that is `EXTERNAL`, all IP_PROTOCOL\n        options other than `ALL` are valid.\n        '
    elif support_l3_default:
        help_str = '        IP protocol that the rule will serve. The default is `TCP`.\n\n        Note that if the load-balancing scheme is `INTERNAL`, the protocol must\n        be one of: `TCP`, `UDP`, `L3_DEFAULT`.\n\n        For a load-balancing scheme that is `EXTERNAL`, all IP_PROTOCOL\n        options are valid.\n        '
    else:
        help_str = '        IP protocol that the rule will serve. The default is `TCP`.\n\n        Note that if the load-balancing scheme is `INTERNAL`, the protocol must\n        be one of: `TCP`, `UDP`.\n\n        For a load-balancing scheme that is `EXTERNAL`, all IP_PROTOCOL\n        options are valid.\n        '
    parser.add_argument('--ip-protocol', choices=protocols, type=lambda x: x.upper(), help=help_str)