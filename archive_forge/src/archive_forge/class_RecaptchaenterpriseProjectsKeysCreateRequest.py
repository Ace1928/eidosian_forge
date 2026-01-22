from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysCreateRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysCreateRequest object.

  Fields:
    googleCloudRecaptchaenterpriseV1Key: A GoogleCloudRecaptchaenterpriseV1Key
      resource to be passed as the request body.
    parent: Required. The name of the project in which the key will be
      created, in the format `projects/{project}`.
  """
    googleCloudRecaptchaenterpriseV1Key = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1Key', 1)
    parent = _messages.StringField(2, required=True)