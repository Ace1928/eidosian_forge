from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UpdateTrackValueValuesEnum(_messages.Enum):
    """Maintenance timing setting: `canary` (Earlier) or `stable` (Later).
    [Learn more](https://cloud.google.com/sql/docs/mysql/instance-
    settings#maintenance-timing-2ndgen).

    Values:
      SQL_UPDATE_TRACK_UNSPECIFIED: This is an unknown maintenance timing
        preference.
      canary: For instance update that requires a restart, this update track
        indicates your instance prefer to restart for new version early in
        maintenance window.
      stable: For instance update that requires a restart, this update track
        indicates your instance prefer to let Cloud SQL choose the timing of
        restart (within its Maintenance window, if applicable).
      week5: For instance update that requires a restart, this update track
        indicates your instance prefer to let Cloud SQL choose the timing of
        restart (within its Maintenance window, if applicable) to be at least
        5 weeks after the notification.
    """
    SQL_UPDATE_TRACK_UNSPECIFIED = 0
    canary = 1
    stable = 2
    week5 = 3