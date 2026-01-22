from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Requested(_messages.Message):
    """An event representing that the Grant was requested.

  Fields:
    expireTime: Output only. The time at which this Grant will expire unless
      the approval workflow completes. If omitted then this request will never
      expire.
  """
    expireTime = _messages.StringField(1)