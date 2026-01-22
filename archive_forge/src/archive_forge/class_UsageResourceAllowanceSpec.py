from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsageResourceAllowanceSpec(_messages.Message):
    """Spec of a usage ResourceAllowance.

  Fields:
    limit: Required. Threshold of a UsageResourceAllowance limiting how many
      resources can be consumed for each type.
    type: Required. Spec type is unique for each usage ResourceAllowance.
      Batch now only supports type as "cpu-core-hours" for CPU usage
      consumption tracking.
  """
    limit = _messages.MessageField('Limit', 1)
    type = _messages.StringField(2)