from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Probe(_messages.Message):
    """Probe describes a health check to be performed against a container to
  determine whether it is alive or ready to receive traffic.

  Fields:
    exec_: Not supported by Cloud Run.
    failureThreshold: Minimum consecutive failures for the probe to be
      considered failed after having succeeded. Defaults to 3. Minimum value
      is 1.
    grpc: GRPCAction specifies an action involving a GRPC port.
    httpGet: HTTPGet specifies the http request to perform.
    initialDelaySeconds: Number of seconds after the container has started
      before the probe is initiated. Defaults to 0 seconds. Minimum value is
      0. Maximum value for liveness probe is 3600. Maximum value for startup
      probe is 240.
    periodSeconds: How often (in seconds) to perform the probe. Default to 10
      seconds. Minimum value is 1. Maximum value for liveness probe is 3600.
      Maximum value for startup probe is 240. Must be greater or equal than
      timeout_seconds.
    successThreshold: Minimum consecutive successes for the probe to be
      considered successful after having failed. Must be 1 if set.
    tcpSocket: TCPSocket specifies an action involving a TCP port.
    timeoutSeconds: Number of seconds after which the probe times out.
      Defaults to 1 second. Minimum value is 1. Maximum value is 3600. Must be
      smaller than period_seconds; if period_seconds is not set, must be less
      or equal than 10.
  """
    exec_ = _messages.MessageField('ExecAction', 1)
    failureThreshold = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    grpc = _messages.MessageField('GRPCAction', 3)
    httpGet = _messages.MessageField('HTTPGetAction', 4)
    initialDelaySeconds = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    periodSeconds = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    successThreshold = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    tcpSocket = _messages.MessageField('TCPSocketAction', 8)
    timeoutSeconds = _messages.IntegerField(9, variant=_messages.Variant.INT32)