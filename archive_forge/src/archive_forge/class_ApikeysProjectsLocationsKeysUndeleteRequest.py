from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsLocationsKeysUndeleteRequest(_messages.Message):
    """A ApikeysProjectsLocationsKeysUndeleteRequest object.

  Fields:
    name: Required. The resource name of the API key to be undeleted.
    v2UndeleteKeyRequest: A V2UndeleteKeyRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    v2UndeleteKeyRequest = _messages.MessageField('V2UndeleteKeyRequest', 2)