from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TCPSocketAction(_messages.Message):
    """TCPSocketAction describes an action based on opening a socket

  Fields:
    host: Not supported by Cloud Run.
    port: Port number to access on the container. Number must be in the range
      1 to 65535.
  """
    host = _messages.StringField(1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)