from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetFutureReservationResourceArg(positional=True):
    if positional:
        name = 'future_reservation'
    else:
        name = '--future-reservation'
    return compute_flags.ResourceArgument(name=name, resource_name='future reservation', completer=ZoneFutureReservationsCompleter, plural=False, required=True, zonal_collection='compute.futureReservations', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION)