from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysUndeleteRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysUndeleteRequest
  object.

  Fields:
    name: Required. The name of the encryption key to undelete.
    undeleteWorkloadIdentityPoolProviderKeyRequest: A
      UndeleteWorkloadIdentityPoolProviderKeyRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteWorkloadIdentityPoolProviderKeyRequest = _messages.MessageField('UndeleteWorkloadIdentityPoolProviderKeyRequest', 2)