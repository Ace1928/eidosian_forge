from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GRPCAction(_messages.Message):
    """GRPCAction describes an action involving a GRPC port.

  Fields:
    port: Port number of the gRPC service. Number must be in the range 1 to
      65535.
    service: Service is the name of the service to place in the gRPC
      HealthCheckRequest. If this is not specified, the default behavior is
      defined by gRPC.
  """
    port = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    service = _messages.StringField(2)