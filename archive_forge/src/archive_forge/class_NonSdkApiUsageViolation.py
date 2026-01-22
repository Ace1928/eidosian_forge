from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NonSdkApiUsageViolation(_messages.Message):
    """Additional details for a non-sdk API usage violation.

  Fields:
    apiSignatures: Signatures of a subset of those hidden API's.
    uniqueApis: Total number of unique hidden API's accessed.
  """
    apiSignatures = _messages.StringField(1, repeated=True)
    uniqueApis = _messages.IntegerField(2, variant=_messages.Variant.INT32)