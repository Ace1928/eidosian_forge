from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysRetrieveLegacySecretKeyRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysRetrieveLegacySecretKeyRequest object.

  Fields:
    key: Required. The public key name linked to the requested secret key in
      the format `projects/{project}/keys/{key}`.
  """
    key = _messages.StringField(1, required=True)