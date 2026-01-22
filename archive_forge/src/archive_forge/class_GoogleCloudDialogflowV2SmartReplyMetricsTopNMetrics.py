from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SmartReplyMetricsTopNMetrics(_messages.Message):
    """Evaluation metrics when retrieving `n` smart replies with the model.

  Fields:
    n: Number of retrieved smart replies. For example, when `n` is 3, this
      evaluation contains metrics for when Dialogflow retrieves 3 smart
      replies with the model.
    recall: Defined as `number of queries whose top n smart replies have at
      least one similar (token match similarity above the defined threshold)
      reply as the real reply` divided by `number of queries with at least one
      smart reply`. Value ranges from 0.0 to 1.0 inclusive.
  """
    n = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    recall = _messages.FloatField(2, variant=_messages.Variant.FLOAT)