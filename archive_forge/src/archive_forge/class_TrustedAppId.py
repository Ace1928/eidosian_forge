from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustedAppId(_messages.Message):
    """JSON template for Trusted App Ids Resource object in Directory API.

  Fields:
    androidPackageName: Android package name.
    certificateHashSHA1: SHA1 signature of the app certificate.
    certificateHashSHA256: SHA256 signature of the app certificate.
    etag: A string attribute.
    kind: Identifies the resource as a trusted AppId.
  """
    androidPackageName = _messages.StringField(1)
    certificateHashSHA1 = _messages.StringField(2)
    certificateHashSHA256 = _messages.StringField(3)
    etag = _messages.StringField(4)
    kind = _messages.StringField(5, default=u'admin#directory#trustedappid')