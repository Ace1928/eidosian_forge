from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TriggerSchedule(_messages.Message):
    """The scan is scheduled to run periodically.

  Fields:
    cron: Required. Cron (https://en.wikipedia.org/wiki/Cron) schedule for
      running scans periodically.To explicitly set a timezone in the cron tab,
      apply a prefix in the cron tab: "CRON_TZ=${IANA_TIME_ZONE}" or
      "TZ=${IANA_TIME_ZONE}". The ${IANA_TIME_ZONE} may only be a valid string
      from IANA time zone database (wikipedia
      (https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List)).
      For example, CRON_TZ=America/New_York 1 * * * *, or TZ=America/New_York
      1 * * * *.This field is required for Schedule scans.
  """
    cron = _messages.StringField(1)