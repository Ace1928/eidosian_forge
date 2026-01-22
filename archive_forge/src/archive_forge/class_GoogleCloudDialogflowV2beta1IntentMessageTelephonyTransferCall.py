from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageTelephonyTransferCall(_messages.Message):
    """Transfers the call in Telephony Gateway.

  Fields:
    phoneNumber: Required. The phone number to transfer the call to in [E.164
      format](https://en.wikipedia.org/wiki/E.164). We currently only allow
      transferring to US numbers (+1xxxyyyzzzz).
  """
    phoneNumber = _messages.StringField(1)