from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableServiceResponse(_messages.Message):
    """Response message for the `EnableService` method. This response message
  is assigned to the `response` field of the returned Operation when that
  operation is done.

  Fields:
    service: The new state of the service after enabling.
  """
    service = _messages.MessageField('GoogleApiServiceusageV1Service', 1)