from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsProvidersPatchRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsProvidersPatchRequest object.

  Fields:
    googleIamV1betaWorkloadIdentityPoolProvider: A
      GoogleIamV1betaWorkloadIdentityPoolProvider resource to be passed as the
      request body.
    name: Output only. The resource name of the provider.
    updateMask: Required. The list of fields to update.
  """
    googleIamV1betaWorkloadIdentityPoolProvider = _messages.MessageField('GoogleIamV1betaWorkloadIdentityPoolProvider', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)