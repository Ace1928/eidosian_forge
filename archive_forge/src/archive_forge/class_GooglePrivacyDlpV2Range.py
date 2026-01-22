from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Range(_messages.Message):
    """Generic half-open interval [start, end)

  Fields:
    end: Index of the last character of the range (exclusive).
    start: Index of the first character of the range (inclusive).
  """
    end = _messages.IntegerField(1)
    start = _messages.IntegerField(2)