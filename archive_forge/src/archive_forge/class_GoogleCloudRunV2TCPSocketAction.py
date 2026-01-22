from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2TCPSocketAction(_messages.Message):
    """TCPSocketAction describes an action based on opening a socket

  Fields:
    port: Optional. Port number to access on the container. Must be in the
      range 1 to 65535. If not specified, defaults to the exposed port of the
      container, which is the value of container.ports[0].containerPort.
  """
    port = _messages.IntegerField(1, variant=_messages.Variant.INT32)