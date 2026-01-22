from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def MaintenanceWindow(sql_messages, instance, maintenance_release_channel=None, maintenance_window_day=None, maintenance_window_hour=None):
    """Generates the maintenance window for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    instance: sql_messages.DatabaseInstance, The original instance, if it might
      be needed to generate the maintenance window.
    maintenance_release_channel: string, which channel's updates to apply.
    maintenance_window_day: string, maintenance window day of week.
    maintenance_window_hour: int, maintenance window hour of day.

  Returns:
    sql_messages.MaintenanceWindow or None

  Raises:
    argparse.ArgumentError: no maintenance window specified.
  """
    channel = maintenance_release_channel
    day = maintenance_window_day
    hour = maintenance_window_hour
    if not any([channel, day, hour]):
        return None
    maintenance_window = sql_messages.MaintenanceWindow(kind='sql#maintenanceWindow')
    if not instance or not instance.settings or (not instance.settings.maintenanceWindow):
        if day is None and hour is not None or (hour is None and day is not None):
            raise argparse.ArgumentError(None, 'There is currently no maintenance window on the instance. To add one, specify values for both day, and hour.')
    if channel:
        names = {'week5': sql_messages.MaintenanceWindow.UpdateTrackValueValuesEnum.week5, 'production': sql_messages.MaintenanceWindow.UpdateTrackValueValuesEnum.stable, 'preview': sql_messages.MaintenanceWindow.UpdateTrackValueValuesEnum.canary}
        maintenance_window.updateTrack = names[channel]
    if day:
        day_num = arg_parsers.DayOfWeek.DAYS.index(day)
        if day_num == 0:
            day_num = 7
        maintenance_window.day = day_num
    if hour is not None:
        maintenance_window.hour = hour
    return maintenance_window