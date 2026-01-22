from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.reservations import util as reservation_util
from googlecloudsdk.core.util import times
def MakeTimeWindowMessage(messages, start_time, end_time, duration):
    """Constructs the time window message object."""
    if end_time:
        return messages.FutureReservationTimeWindow(startTime=start_time, endTime=end_time)
    else:
        return messages.FutureReservationTimeWindow(startTime=start_time, duration=messages.Duration(seconds=duration))