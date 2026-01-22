from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsSendVerificationCodeRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsSendVerificationCodeRequest
  object.

  Fields:
    name: Required. The notification channel to which to send a verification
      code.
    sendNotificationChannelVerificationCodeRequest: A
      SendNotificationChannelVerificationCodeRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    sendNotificationChannelVerificationCodeRequest = _messages.MessageField('SendNotificationChannelVerificationCodeRequest', 2)