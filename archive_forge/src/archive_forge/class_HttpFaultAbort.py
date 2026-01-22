from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpFaultAbort(_messages.Message):
    """Specification for how requests are aborted as part of fault injection.

  Fields:
    httpStatus: The HTTP status code used to abort the request. The value must
      be from 200 to 599 inclusive. For gRPC protocol, the gRPC status code is
      mapped to HTTP status code according to this mapping table. HTTP status
      200 is mapped to gRPC status UNKNOWN. Injecting an OK status is
      currently not supported by Traffic Director.
    percentage: The percentage of traffic for connections, operations, or
      requests that is aborted as part of fault injection. The value must be
      from 0.0 to 100.0 inclusive.
  """
    httpStatus = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    percentage = _messages.FloatField(2)