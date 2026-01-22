from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1MaintenanceSchedule(_messages.Message):
    """Maintenance schedule which is exposed to customer and potentially end
  user, indicating published upcoming future maintenance schedule

  Fields:
    canReschedule: This field is deprecated, and will be always set to true
      since reschedule can happen multiple times now. This field should not be
      removed until all service producers remove this for their customers.
    endTime: The scheduled end time for the maintenance.
    rolloutManagementPolicy: The rollout management policy this maintenance
      schedule is associated with. When doing reschedule update request, the
      reschedule should be against this given policy.
    scheduleDeadlineTime: schedule_deadline_time is the time deadline any
      schedule start time cannot go beyond, including reschedule. It's
      normally the initial schedule start time plus maintenance window length
      (1 day or 1 week). Maintenance cannot be scheduled to start beyond this
      deadline.
    startTime: The scheduled start time for the maintenance.
  """
    canReschedule = _messages.BooleanField(1)
    endTime = _messages.StringField(2)
    rolloutManagementPolicy = _messages.StringField(3)
    scheduleDeadlineTime = _messages.StringField(4)
    startTime = _messages.StringField(5)