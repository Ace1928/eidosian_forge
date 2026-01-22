from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysMigrateRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysMigrateRequest object.

  Fields:
    googleCloudRecaptchaenterpriseV1MigrateKeyRequest: A
      GoogleCloudRecaptchaenterpriseV1MigrateKeyRequest resource to be passed
      as the request body.
    name: Required. The name of the key to be migrated, in the format
      `projects/{project}/keys/{key}`.
  """
    googleCloudRecaptchaenterpriseV1MigrateKeyRequest = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1MigrateKeyRequest', 1)
    name = _messages.StringField(2, required=True)