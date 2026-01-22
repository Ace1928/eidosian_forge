from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VerificationCode(_messages.Message):
    """JSON template for verification codes in Directory API.

  Fields:
    etag: ETag of the resource.
    kind: The type of the resource. This is always
      admin#directory#verificationCode.
    userId: The obfuscated unique ID of the user.
    verificationCode: A current verification code for the user. Invalidated or
      used verification codes are not returned as part of the result.
  """
    etag = _messages.StringField(1)
    kind = _messages.StringField(2, default=u'admin#directory#verificationCode')
    userId = _messages.StringField(3)
    verificationCode = _messages.StringField(4)