from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesUndeleteRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesUn
  deleteRequest object.

  Fields:
    name: Required. The name of the managed identity to undelete.
    undeleteWorkloadIdentityPoolManagedIdentityRequest: A
      UndeleteWorkloadIdentityPoolManagedIdentityRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteWorkloadIdentityPoolManagedIdentityRequest = _messages.MessageField('UndeleteWorkloadIdentityPoolManagedIdentityRequest', 2)