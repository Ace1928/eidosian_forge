from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import flags as r_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
def _AutoDeleteUpdateRequest(args, reservation_ref, holder):
    """Create Update Request for mofigying auto-delete properties."""
    messages = holder.client.messages
    update_mask = []
    if args.IsSpecified('delete_at_time'):
        update_mask.append('deleteAtTime')
        delete_at_time = args.delete_at_time
    else:
        delete_at_time = None
    if args.IsSpecified('delete_after_duration'):
        update_mask.append('deleteAfterDuration')
        delete_after_duration = args.delete_after_duration
    else:
        delete_after_duration = None
    if args.IsSpecified('disable_auto_delete'):
        update_mask.append('deleteAtTime')
    r_resource = util.MakeReservationMessage(messages, reservation_ref.Name(), None, None, None, None, reservation_ref.zone, delete_at_time, delete_after_duration)
    return messages.ComputeReservationsUpdateRequest(reservation=reservation_ref.Name(), reservationResource=r_resource, paths=update_mask, project=reservation_ref.project, zone=reservation_ref.zone)