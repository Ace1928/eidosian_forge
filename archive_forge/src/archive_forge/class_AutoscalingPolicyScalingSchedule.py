from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingPolicyScalingSchedule(_messages.Message):
    """Scaling based on user-defined schedule. The message describes a single
  scaling schedule. A scaling schedule changes the minimum number of VM
  instances an autoscaler can recommend, which can trigger scaling out.

  Fields:
    description: A description of a scaling schedule.
    disabled: A boolean value that specifies whether a scaling schedule can
      influence autoscaler recommendations. If set to true, then a scaling
      schedule has no effect. This field is optional, and its value is false
      by default.
    durationSec: The duration of time intervals, in seconds, for which this
      scaling schedule is to run. The minimum allowed value is 300. This field
      is required.
    minRequiredReplicas: The minimum number of VM instances that the
      autoscaler will recommend in time intervals starting according to
      schedule. This field is required.
    schedule: The start timestamps of time intervals when this scaling
      schedule is to provide a scaling signal. This field uses the extended
      cron format (with an optional year field). The expression can describe a
      single timestamp if the optional year is set, in which case the scaling
      schedule runs once. The schedule is interpreted with respect to
      time_zone. This field is required. Note: These timestamps only describe
      when autoscaler starts providing the scaling signal. The VMs need
      additional time to become serving.
    timeZone: The time zone to use when interpreting the schedule. The value
      of this field must be a time zone name from the tz database:
      https://en.wikipedia.org/wiki/Tz_database. This field is assigned a
      default value of "UTC" if left empty.
  """
    description = _messages.StringField(1)
    disabled = _messages.BooleanField(2)
    durationSec = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    minRequiredReplicas = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    schedule = _messages.StringField(5)
    timeZone = _messages.StringField(6)