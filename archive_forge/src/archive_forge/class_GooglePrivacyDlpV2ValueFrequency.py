from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ValueFrequency(_messages.Message):
    """A value of a field, including its frequency.

  Fields:
    count: How many times the value is contained in the field.
    value: A value contained in the field in question.
  """
    count = _messages.IntegerField(1)
    value = _messages.MessageField('GooglePrivacyDlpV2Value', 2)