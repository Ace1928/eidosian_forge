from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ResponseMessageTelephonyTransferCall(_messages.Message):
    """Represents the signal that telles the client to transfer the phone call
  connected to the agent to a third-party endpoint.

  Fields:
    phoneNumber: Transfer the call to a phone number in [E.164
      format](https://en.wikipedia.org/wiki/E.164).
  """
    phoneNumber = _messages.StringField(1)