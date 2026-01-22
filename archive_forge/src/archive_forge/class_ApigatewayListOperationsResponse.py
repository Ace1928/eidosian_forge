from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayListOperationsResponse(_messages.Message):
    """The response message for Operations.ListOperations.

  Fields:
    nextPageToken: The standard List next-page token.
    operations: A list of operations that matches the specified filter in the
      request.
  """
    nextPageToken = _messages.StringField(1)
    operations = _messages.MessageField('ApigatewayOperation', 2, repeated=True)