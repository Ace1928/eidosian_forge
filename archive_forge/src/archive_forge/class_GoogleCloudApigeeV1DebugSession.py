from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DebugSession(_messages.Message):
    """A GoogleCloudApigeeV1DebugSession object.

  Fields:
    count: Optional. The number of request to be traced. Min = 1, Max = 15,
      Default = 10.
    createTime: Output only. The first transaction creation timestamp,
      recorded by UAP.
    filter: Optional. A conditional statement which is evaluated against the
      request message to determine if it should be traced. Syntax matches that
      of on API Proxy bundle flow Condition.
    name: A unique ID for this DebugSession.
    timeout: Optional. The time in seconds after which this DebugSession
      should end. This value will override the value in query param, if both
      are provided.
    tracesize: Optional. The maximum number of bytes captured from the
      response payload. Min = 0, Max = 5120, Default = 5120.
    validity: Optional. The length of time, in seconds, that this debug
      session is valid, starting from when it's received in the control plane.
      Min = 1, Max = 15, Default = 10.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    createTime = _messages.StringField(2)
    filter = _messages.StringField(3)
    name = _messages.StringField(4)
    timeout = _messages.IntegerField(5)
    tracesize = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    validity = _messages.IntegerField(7, variant=_messages.Variant.INT32)