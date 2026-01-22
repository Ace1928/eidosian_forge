from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkloadIdentityPoolManagedIdentitiesResponse(_messages.Message):
    """Response message for ListWorkloadIdentityPoolManagedIdentities.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    workloadIdentityPoolManagedIdentities: A list of managed identities.
  """
    nextPageToken = _messages.StringField(1)
    workloadIdentityPoolManagedIdentities = _messages.MessageField('WorkloadIdentityPoolManagedIdentity', 2, repeated=True)