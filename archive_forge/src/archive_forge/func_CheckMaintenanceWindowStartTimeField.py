from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
import six
def CheckMaintenanceWindowStartTimeField(maintenance_window_start_time):
    if maintenance_window_start_time < 0 or maintenance_window_start_time > 23:
        raise InvalidTimeOfDayError('A valid time of day must be specified (0, 23) hours.')
    return maintenance_window_start_time