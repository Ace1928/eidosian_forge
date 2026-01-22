from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetReservationResourceArg(positional=True):
    if positional:
        name = 'reservation'
    else:
        name = '--reservation'
    return compute_flags.ResourceArgument(name=name, resource_name='reservation', completer=ZoneReservationsCompleter, plural=False, required=True, zonal_collection='compute.reservations', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION)