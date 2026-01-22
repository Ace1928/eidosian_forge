from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileResultProfileFieldProfileInfoStringFieldInfo(_messages.Message):
    """The profile information for a string type field.

  Fields:
    averageLength: Average length of non-null values in the scanned data.
    maxLength: Maximum length of non-null values in the scanned data.
    minLength: Minimum length of non-null values in the scanned data.
  """
    averageLength = _messages.FloatField(1)
    maxLength = _messages.IntegerField(2)
    minLength = _messages.IntegerField(3)