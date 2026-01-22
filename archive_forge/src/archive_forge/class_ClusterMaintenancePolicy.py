from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterMaintenancePolicy(_messages.Message):
    """Maintenance policy per cluster.

  Fields:
    createTime: Output only. The time when the policy was created i.e.
      Maintenance Window or Deny Period was assigned.
    denyMaintenancePeriods: Deny maintenance periods
    updateTime: Output only. The time when the policy was updated i.e.
      Maintenance Window or Deny Period was updated.
    weeklyMaintenanceWindow: Optional. Maintenance window that is applied to
      resources covered by this policy. Minimum 1. For the current version,
      the maximum number of weekly_maintenance_window is expected to be one.
  """
    createTime = _messages.StringField(1)
    denyMaintenancePeriods = _messages.MessageField('ClusterDenyMaintenancePeriod', 2, repeated=True)
    updateTime = _messages.StringField(3)
    weeklyMaintenanceWindow = _messages.MessageField('ClusterWeeklyMaintenanceWindow', 4, repeated=True)