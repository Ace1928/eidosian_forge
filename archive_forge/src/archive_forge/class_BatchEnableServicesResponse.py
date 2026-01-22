from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchEnableServicesResponse(_messages.Message):
    """Response message for the `BatchEnableServices` method. This response
  message is assigned to the `response` field of the returned Operation when
  that operation is done.

  Fields:
    failures: If allow_partial_success is true, and one or more services could
      not be enabled, this field contains the details about each failure.
    services: The new state of the services after enabling.
  """
    failures = _messages.MessageField('EnableFailure', 1, repeated=True)
    services = _messages.MessageField('GoogleApiServiceusageV1Service', 2, repeated=True)