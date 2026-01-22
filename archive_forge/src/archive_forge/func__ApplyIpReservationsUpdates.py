from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bms.bms_client import BmsClient
from googlecloudsdk.api_lib.bms.bms_client import IpRangeReservation
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bms import exceptions
from googlecloudsdk.command_lib.bms import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _ApplyIpReservationsUpdates(args, existing_network):
    """Applies the changes in args to the reservations in existing_network.

  Returns None if no changes were to be applied.

  Args:
    args: The arguments passed to the command.
    existing_network: The existing network.

  Returns:
    List of IP range reservations after applying updates or None if there are
    no changes.
  """
    if _IsSpecified(args, 'clear_ip_range_reservations'):
        return []
    existing_reservations = [IpRangeReservation(res.startAddress, res.endAddress, res.note) for res in existing_network.reservations]
    if _IsSpecified(args, 'add_ip_range_reservation'):
        res_dict = args.add_ip_range_reservation
        _ValidateAgainstSpec(res_dict, flags.IP_RESERVATION_SPEC, 'add-ip-range-reservation')
        return existing_reservations + [IpRangeReservation(res_dict['start-address'], res_dict['end-address'], res_dict['note'])]
    if _IsSpecified(args, 'remove_ip_range_reservation'):
        return _RemoveReservation(existing_reservations, args.remove_ip_range_reservation)