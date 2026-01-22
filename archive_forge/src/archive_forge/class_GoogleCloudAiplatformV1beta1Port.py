from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Port(_messages.Message):
    """Represents a network port in a container.

  Fields:
    containerPort: The number of the port to expose on the pod's IP address.
      Must be a valid port number, between 1 and 65535 inclusive.
  """
    containerPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)