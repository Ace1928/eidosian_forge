from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
from googlecloudsdk.core import yaml
def MakeSourceDestReservations(args, messages):
    """Return messages required for update-reservations command."""
    source_msg = ReservationArgToMessage('source_reservation', 'source_accelerator', 'source_local_ssd', 'source_share_setting', 'source_share_with', args, messages)
    destination_msg = ReservationArgToMessage('dest_reservation', 'dest_accelerator', 'dest_local_ssd', 'dest_share_setting', 'dest_share_with', args, messages)
    return [source_msg, destination_msg]