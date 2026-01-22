from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyInstanceSchedulePolicySchedule(_messages.Message):
    """Schedule for an instance operation.

  Fields:
    schedule: Specifies the frequency for the operation, using the unix-cron
      format.
  """
    schedule = _messages.StringField(1)