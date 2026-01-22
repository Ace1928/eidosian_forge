from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentHandoffConfigLivePersonConfig(_messages.Message):
    """Configuration specific to LivePerson (https://www.liveperson.com).

  Fields:
    accountNumber: Required. Account number of the LivePerson account to
      connect. This is the account number you input at the login page.
  """
    accountNumber = _messages.StringField(1)