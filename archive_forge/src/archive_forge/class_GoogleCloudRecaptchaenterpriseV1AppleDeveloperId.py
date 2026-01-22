from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1AppleDeveloperId(_messages.Message):
    """Contains fields that are required to perform Apple-specific integrity
  checks.

  Fields:
    keyId: Required. The Apple developer key ID (10-character string).
    privateKey: Required. Input only. A private key (downloaded as a text file
      with a .p8 file extension) generated for your Apple Developer account.
      Ensure that Apple DeviceCheck is enabled for the private key.
    teamId: Required. The Apple team ID (10-character string) owning the
      provisioning profile used to build your application.
  """
    keyId = _messages.StringField(1)
    privateKey = _messages.StringField(2)
    teamId = _messages.StringField(3)