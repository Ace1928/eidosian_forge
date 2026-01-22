from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsProvidersUndeleteRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsProvidersUndeleteRequest
  object.

  Fields:
    googleIamV1betaUndeleteWorkloadIdentityPoolProviderRequest: A
      GoogleIamV1betaUndeleteWorkloadIdentityPoolProviderRequest resource to
      be passed as the request body.
    name: Required. The name of the provider to undelete.
  """
    googleIamV1betaUndeleteWorkloadIdentityPoolProviderRequest = _messages.MessageField('GoogleIamV1betaUndeleteWorkloadIdentityPoolProviderRequest', 1)
    name = _messages.StringField(2, required=True)