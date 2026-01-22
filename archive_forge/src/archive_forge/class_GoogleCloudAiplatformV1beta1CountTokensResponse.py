from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CountTokensResponse(_messages.Message):
    """Response message for PredictionService.CountTokens.

  Fields:
    totalBillableCharacters: The total number of billable characters counted
      across all instances from the request.
    totalTokens: The total number of tokens counted across all instances from
      the request.
  """
    totalBillableCharacters = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    totalTokens = _messages.IntegerField(2, variant=_messages.Variant.INT32)