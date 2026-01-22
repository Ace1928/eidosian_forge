from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KMapEstimationQuasiIdValues(_messages.Message):
    """A tuple of values for the quasi-identifier columns.

  Fields:
    estimatedAnonymity: The estimated anonymity for these quasi-identifier
      values.
    quasiIdsValues: The quasi-identifier values.
  """
    estimatedAnonymity = _messages.IntegerField(1)
    quasiIdsValues = _messages.MessageField('GooglePrivacyDlpV2Value', 2, repeated=True)