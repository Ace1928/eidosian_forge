from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Probe(_messages.Message):
    """Probe describes a health check to be performed against a container to
  determine whether it is alive or ready to receive traffic.

  Fields:
    exec_: Exec specifies the action to take.
    periodSeconds: How often (in seconds) to perform the probe. Default to 10
      seconds. Minimum value is 1. Must be less than timeout_seconds. Maps to
      Kubernetes probe argument 'periodSeconds'.
    timeoutSeconds: Number of seconds after which the probe times out.
      Defaults to 1 second. Minimum value is 1. Must be greater or equal to
      period_seconds. Maps to Kubernetes probe argument 'timeoutSeconds'.
  """
    exec_ = _messages.MessageField('GoogleCloudAiplatformV1ProbeExecAction', 1)
    periodSeconds = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    timeoutSeconds = _messages.IntegerField(3, variant=_messages.Variant.INT32)