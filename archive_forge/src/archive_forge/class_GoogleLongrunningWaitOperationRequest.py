from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleLongrunningWaitOperationRequest(_messages.Message):
    """The request message for Operations.WaitOperation.

  Fields:
    timeout: The maximum duration to wait before timing out. If left blank,
      the wait will be at most the time permitted by the underlying HTTP/RPC
      protocol. If RPC context deadline is also specified, the shorter one
      will be used.
  """
    timeout = _messages.StringField(1)