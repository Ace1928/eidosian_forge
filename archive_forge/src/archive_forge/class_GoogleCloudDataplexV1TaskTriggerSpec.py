from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskTriggerSpec(_messages.Message):
    """Task scheduling and trigger settings.

  Enums:
    TypeValueValuesEnum: Required. Immutable. Trigger type of the user-
      specified Task.

  Fields:
    disabled: Optional. Prevent the task from executing. This does not cancel
      already running tasks. It is intended to temporarily disable RECURRING
      tasks.
    maxRetries: Optional. Number of retry attempts before aborting. Set to
      zero to never attempt to retry a failed task.
    schedule: Optional. Cron schedule (https://en.wikipedia.org/wiki/Cron) for
      running tasks periodically. To explicitly set a timezone to the cron
      tab, apply a prefix in the cron tab: "CRON_TZ=${IANA_TIME_ZONE}" or
      "TZ=${IANA_TIME_ZONE}". The ${IANA_TIME_ZONE} may only be a valid string
      from IANA time zone database. For example, CRON_TZ=America/New_York 1 *
      * * *, or TZ=America/New_York 1 * * * *. This field is required for
      RECURRING tasks.
    startTime: Optional. The first run of the task will be after this time. If
      not specified, the task will run shortly after being submitted if
      ON_DEMAND and based on the schedule if RECURRING.
    type: Required. Immutable. Trigger type of the user-specified Task.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. Trigger type of the user-specified Task.

    Values:
      TYPE_UNSPECIFIED: Unspecified trigger type.
      ON_DEMAND: The task runs one-time shortly after Task Creation.
      RECURRING: The task is scheduled to run periodically.
    """
        TYPE_UNSPECIFIED = 0
        ON_DEMAND = 1
        RECURRING = 2
    disabled = _messages.BooleanField(1)
    maxRetries = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    schedule = _messages.StringField(3)
    startTime = _messages.StringField(4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)