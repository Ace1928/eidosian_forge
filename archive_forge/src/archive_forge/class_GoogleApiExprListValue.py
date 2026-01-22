from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprListValue(_messages.Message):
    """A list. Wrapped in a message so 'not set' and empty can be
  differentiated, which is required for use in a 'oneof'.

  Fields:
    values: The ordered values in the list.
  """
    values = _messages.MessageField('GoogleApiExprValue', 1, repeated=True)