from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CrontabSpec(_messages.Message):
    """CrontabSpec can be used to specify the version timestamp and frequency
  at which the backup should be created.

  Fields:
    creationWindow: Output only. Schedule backups will contain an externally
      consistent copy of the database at the version timestamp specified in
      `schedule_spec.cron_spec`. However, Spanner may not initiate the
      creation of the scheduled backups at that version timestamp. Spanner
      will initiate the creation of scheduled backups within the time window
      bounded by the version_time specified in `schedule_spec.cron_spec` and
      version_time + `creation_window`.
    text: Required. Textual representation of the crontab. User can customize
      the backup frequency and the backup version timestamp using the cron
      expression. The version timestamp must be in UTC timzeone. The backup
      will contain an externally consistent copy of the database at the
      version timestamp. Allowed frequencies are 12 hour, 1 day, 1 week and 1
      month. Examples of valid cron specifications: * `0 2/12 * * * ` : every
      12 hours at (2, 14) hours past midnight in UTC. * `0 2,14 * * * ` :
      every 12 hours at (2,14) hours past midnight in UTC. * `0 2 * * * ` :
      once a day at 2 past midnight in UTC. * `0 2 * * 0 ` : once a week every
      Sunday at 2 past midnight in UTC. * `0 2 8 * * ` : once a month on 8th
      day at 2 past midnight in UTC.
    timeZone: Output only. The time zone of the times in `CrontabSpec.text`.
      Currently only UTC is supported.
  """
    creationWindow = _messages.StringField(1)
    text = _messages.StringField(2)
    timeZone = _messages.StringField(3)