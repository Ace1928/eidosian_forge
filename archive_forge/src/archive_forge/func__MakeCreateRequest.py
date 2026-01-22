from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.future_reservations import flags
from googlecloudsdk.command_lib.compute.future_reservations import resource_args
from googlecloudsdk.command_lib.compute.future_reservations import util
def _MakeCreateRequest(args, messages, resources, project, future_reservation_ref):
    """Common routine for creating future reservation request."""
    future_reservation = util.MakeFutureReservationMessageFromArgs(messages, resources, args, future_reservation_ref)
    future_reservation.description = args.description
    future_reservation.namePrefix = args.name_prefix
    return messages.ComputeFutureReservationsInsertRequest(futureReservation=future_reservation, project=project, zone=future_reservation_ref.zone)