from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _ConstructMaintenanceWindowPatch(maintenance_window_start, maintenance_window_end, maintenance_window_recurrence, clear_maintenance_window, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch for updating maintenance window.

  Args:
    maintenance_window_start: Datetime or None, a starting date of the
      maintenance window.
    maintenance_window_end: Datetime or None, an ending date of the maintenance
      window.
    maintenance_window_recurrence: str or None, recurrence RRULE for the
      maintenance window.
    clear_maintenance_window: bool or None, specifies if maintenance window
      options should be cleared.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    if clear_maintenance_window:
        return ('config.maintenance_window', messages.Environment())
    window_value = messages.MaintenanceWindow(startTime=maintenance_window_start.isoformat(), endTime=maintenance_window_end.isoformat(), recurrence=maintenance_window_recurrence)
    config = messages.EnvironmentConfig(maintenanceWindow=window_value)
    return ('config.maintenance_window', messages.Environment(config=config))