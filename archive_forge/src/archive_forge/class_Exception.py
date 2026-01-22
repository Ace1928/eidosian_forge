from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Exception(_messages.Message):
    """Exception describes why the step entry failed.

  Fields:
    payload: Error message represented as a JSON string.
  """
    payload = _messages.StringField(1)