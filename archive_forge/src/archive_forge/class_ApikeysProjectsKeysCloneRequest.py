from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsKeysCloneRequest(_messages.Message):
    """A ApikeysProjectsKeysCloneRequest object.

  Fields:
    name: Required. The resource name of the Api key to be cloned under same
      parent. `apikeys.keys.get permission` and `apikeys.keys.create
      permission` are required for parent resource.
    v2alpha1CloneKeyRequest: A V2alpha1CloneKeyRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    v2alpha1CloneKeyRequest = _messages.MessageField('V2alpha1CloneKeyRequest', 2)