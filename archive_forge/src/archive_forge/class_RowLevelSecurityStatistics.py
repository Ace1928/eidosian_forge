from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RowLevelSecurityStatistics(_messages.Message):
    """Statistics for row-level security.

  Fields:
    rowLevelSecurityApplied: Whether any accessed data was protected by row
      access policies.
  """
    rowLevelSecurityApplied = _messages.BooleanField(1)