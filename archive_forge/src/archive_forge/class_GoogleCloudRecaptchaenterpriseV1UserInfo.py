from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1UserInfo(_messages.Message):
    """User information associated with a request protected by reCAPTCHA
  Enterprise.

  Fields:
    accountId: Optional. For logged-in requests or login/registration
      requests, the unique account identifier associated with this user. You
      can use the username if it is stable (meaning it is the same for every
      request associated with the same user), or any stable user ID of your
      choice. Leave blank for non logged-in actions or guest checkout.
    createAccountTime: Optional. Creation time for this account associated
      with this user. Leave blank for non logged-in actions, guest checkout,
      or when there is no account associated with the current user.
    userIds: Optional. Identifiers associated with this user or request.
  """
    accountId = _messages.StringField(1)
    createAccountTime = _messages.StringField(2)
    userIds = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1UserId', 3, repeated=True)