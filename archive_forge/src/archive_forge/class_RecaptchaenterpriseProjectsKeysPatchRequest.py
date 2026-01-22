from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsKeysPatchRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsKeysPatchRequest object.

  Fields:
    googleCloudRecaptchaenterpriseV1Key: A GoogleCloudRecaptchaenterpriseV1Key
      resource to be passed as the request body.
    name: Identifier. The resource name for the Key in the format
      `projects/{project}/keys/{key}`.
    updateMask: Optional. The mask to control which fields of the key get
      updated. If the mask is not present, all fields will be updated.
  """
    googleCloudRecaptchaenterpriseV1Key = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1Key', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)