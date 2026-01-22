from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorLogEntry(_messages.Message):
    """An entry describing an error that has occurred.

  Fields:
    errorDetails: A list of messages that carry the error details.
    url: Required. A URL that refers to the target (a data source, a data
      sink, or an object) with which the error is associated.
  """
    errorDetails = _messages.StringField(1, repeated=True)
    url = _messages.StringField(2)