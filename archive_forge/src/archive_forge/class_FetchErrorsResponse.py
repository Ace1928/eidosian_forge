from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchErrorsResponse(_messages.Message):
    """Response message for a 'FetchErrors' response.

  Fields:
    errors: The list of errors on the Stream.
  """
    errors = _messages.MessageField('Error', 1, repeated=True)