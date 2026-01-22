from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def CheckTimeOfDayField(time_of_day, error_message, arg):
    """Check if input is a valid TimeOfDay format."""
    hour_and_min = time_of_day.split(':')
    if len(hour_and_min) != 2 or not hour_and_min[0].isdigit() or (not hour_and_min[1].isdigit()):
        raise exceptions.InvalidArgumentException(arg, error_message)
    hour = int(hour_and_min[0])
    minute = int(hour_and_min[1])
    if hour < 0 or minute < 0 or hour > 23 or (minute > 59):
        raise exceptions.InvalidArgumentException(arg, error_message)