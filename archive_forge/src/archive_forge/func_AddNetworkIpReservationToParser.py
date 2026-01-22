from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNetworkIpReservationToParser(parser, hidden):
    """Adds the flags for network IP range reservation to parser."""
    group_arg = parser.add_mutually_exclusive_group(required=False)
    group_arg.add_argument('--add-ip-range-reservation', type=arg_parsers.ArgDict(spec=IP_RESERVATION_SPEC), metavar='PROPERTY=VALUE', help='\n              Add a reservation of a range of IP addresses in the network.\n\n              *start_address*::: The first address of this reservation block.\n              Must be specified as a single IPv4 address, e.g. `10.1.2.2`.\n\n              *end_address*::: The last address of this reservation block,\n              inclusive. I.e., for cases when reservations are only single\n              addresses, end_address and start_address will be the same.\n              Must be specified as a single IPv4 address, e.g. `10.1.2.2`.\n\n              *note*::: A note about this reservation, intended for human consumption.\n            ', hidden=hidden)
    group_arg.add_argument('--remove-ip-range-reservation', type=arg_parsers.ArgDict(spec=IP_RESERVATION_KEY_SPEC), metavar='PROPERTY=VALUE', help='\n              Remove a reservation of a range of IP addresses in the network.\n\n              *start_address*::: The first address of the reservation block to remove.\n\n              *end_address*::: The last address of the reservation block to remove.\n            ', hidden=hidden)
    group_arg.add_argument('--clear-ip-range-reservations', action='store_true', help='Removes all IP range reservations in the network.', hidden=hidden)