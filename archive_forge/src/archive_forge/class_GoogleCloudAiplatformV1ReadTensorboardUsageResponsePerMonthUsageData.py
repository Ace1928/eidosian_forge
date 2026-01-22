from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ReadTensorboardUsageResponsePerMonthUsageData(_messages.Message):
    """Per month usage data

  Fields:
    userUsageData: Usage data for each user in the given month.
  """
    userUsageData = _messages.MessageField('GoogleCloudAiplatformV1ReadTensorboardUsageResponsePerUserUsageData', 1, repeated=True)