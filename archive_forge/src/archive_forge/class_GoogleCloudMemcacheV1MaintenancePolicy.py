from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMemcacheV1MaintenancePolicy(_messages.Message):
    """Maintenance policy per instance.

  Fields:
    createTime: Output only. The time when the policy was created.
    description: Description of what this policy is for. Create/Update methods
      return INVALID_ARGUMENT if the length is greater than 512.
    updateTime: Output only. The time when the policy was updated.
    weeklyMaintenanceWindow: Required. Maintenance window that is applied to
      resources covered by this policy. Minimum 1. For the current version,
      the maximum number of weekly_maintenance_windows is expected to be one.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    updateTime = _messages.StringField(3)
    weeklyMaintenanceWindow = _messages.MessageField('WeeklyMaintenanceWindow', 4, repeated=True)