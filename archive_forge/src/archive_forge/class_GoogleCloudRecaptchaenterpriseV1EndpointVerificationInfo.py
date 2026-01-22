from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1EndpointVerificationInfo(_messages.Message):
    """Information about a verification endpoint that can be used for 2FA.

  Fields:
    emailAddress: Email address for which to trigger a verification request.
    lastVerificationTime: Output only. Timestamp of the last successful
      verification for the endpoint, if any.
    phoneNumber: Phone number for which to trigger a verification request.
      Should be given in E.164 format.
    requestToken: Output only. Token to provide to the client to trigger
      endpoint verification. It must be used within 15 minutes.
  """
    emailAddress = _messages.StringField(1)
    lastVerificationTime = _messages.StringField(2)
    phoneNumber = _messages.StringField(3)
    requestToken = _messages.StringField(4)