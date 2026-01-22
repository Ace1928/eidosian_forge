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
def AddressArgHelp():
    """Build the help text for the address argument."""
    lb_schemes = '(EXTERNAL, EXTERNAL_MANAGED, INTERNAL, INTERNAL_SELF_MANAGED, INTERNAL_MANAGED)'
    detailed_help = "    The IP address that the forwarding rule serves. When a client sends traffic\n    to this IP address, the forwarding rule directs the traffic to the\n    target that you specify in the forwarding rule.\n\n    If you don't specify a reserved IP address, an ephemeral IP address is\n    assigned. You can specify the IP address as a literal IP address or as a\n    reference to an existing Address resource. The following examples are\n    all valid:\n    * 100.1.2.3\n    * 2600:1901::/96\n    * https://compute.googleapis.com/compute/v1/projects/project-1/regions/us-central1/addresses/address-1\n    * projects/project-1/regions/us-central1/addresses/address-1\n    * regions/us-central1/addresses/address-1\n    * global/addresses/address-1\n    * address-1\n\n    The load-balancing-scheme %s and the target of the forwarding rule\n    determine the type of IP address that you can use. The address\n    type must be external for load-balancing-scheme EXTERNAL or\n    EXTERNAL_MANAGED. For other load-balancing-schemes, the address type\n    must be internal. For detailed information, refer to\n    https://cloud.google.com/load-balancing/docs/forwarding-rule-concepts#ip_address_specifications.\n  " % lb_schemes
    return textwrap.dedent(detailed_help)