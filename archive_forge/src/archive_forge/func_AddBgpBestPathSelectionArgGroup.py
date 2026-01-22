from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddBgpBestPathSelectionArgGroup(parser):
    """Adds the BGP Best Path Selection flags.

  Args:
    parser (parser_arguments.ArgumentInterceptor): Argument parser
  """
    group = parser.add_argument_group(help='BGP Best Path Selection flags')
    group.add_argument('--bgp-best-path-selection-mode', choices={'LEGACY': 'Dynamic routes are ranked based on MED BGP attribute. When global routing is enabled, MED of the routes received from other regions is original MED plus region-to-region cost.', 'STANDARD': 'Dynamic routes are ranked based on AS Path, Origin, Neighbor ASN and MED BGP attributes. When global routing is enabled, region-to-region cost is used as a tiebreaker. This mode offers customizations to fine-tune BGP best path routing with additional knobs like --bgp-bps-always-compare-med and --bgp-bps-inter-region-cost'}, help='The BGP best selection algorithm to be employed. MODE can be LEGACY or STANDARD.', type=arg_utils.ChoiceToEnumName)
    group.add_argument('--bgp-bps-always-compare-med', action=arg_parsers.StoreTrueFalseAction, help='Enables/disables the comparison of MED across routes with different NeighborAsn. This value can only be set if the --bgp-best-path-selection-mode is STANDARD.')
    group.add_argument('--bgp-bps-inter-region-cost', choices={'DEFAULT': 'MED is compared as originally received from peers. Cost is evaluated as a next step when MED is the same.', 'ADD_COST_TO_MED': 'Adds inter-region cost to the MED before comparing MED value. When multiple routes have the same value after the Add-cost-to-med comparison, the route selection continues and prefers the route with lowest cost.'}, help='Allows to define preferred approach for handling inter-region cost in the selection process. This value can only be set if the --bgp-best-path-selection-mode is STANDARD.', type=arg_utils.ChoiceToEnumName)