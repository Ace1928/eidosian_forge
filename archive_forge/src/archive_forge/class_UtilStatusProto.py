from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UtilStatusProto(_messages.Message):
    """Wire-format for a Status object

  Fields:
    canonicalCode: The canonical error code (see codes.proto) that most
      closely corresponds to this status. This may be missing, and in the
      common case of the generic space, it definitely will be.
    code: Numeric code drawn from the space specified below. Often, this is
      the canonical error space, and code is drawn from
      google3/util/task/codes.proto
    message: Detail message
    messageSet: message_set associates an arbitrary proto message with the
      status.
    space: The following are usually only present when code != 0 Space to
      which this status belongs
  """
    canonicalCode = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    code = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    message = _messages.StringField(3)
    messageSet = _messages.MessageField('Proto2BridgeMessageSet', 4)
    space = _messages.StringField(5)