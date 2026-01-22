from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityActionDeny(_messages.Message):
    """Message that should be set in case of a Deny Action.

  Fields:
    responseCode: Optional. The HTTP response code if the Action = DENY.
  """
    responseCode = _messages.IntegerField(1, variant=_messages.Variant.INT32)