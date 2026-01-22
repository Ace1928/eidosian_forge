from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanExecutionStatus(_messages.Message):
    """Status of the data scan execution.

  Fields:
    latestJobEndTime: The time when the latest DataScanJob ended.
    latestJobStartTime: The time when the latest DataScanJob started.
  """
    latestJobEndTime = _messages.StringField(1)
    latestJobStartTime = _messages.StringField(2)