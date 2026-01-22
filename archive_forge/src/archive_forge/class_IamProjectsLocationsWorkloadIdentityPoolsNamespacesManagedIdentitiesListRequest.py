from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesListRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesLi
  stRequest object.

  Fields:
    pageSize: The maximum number of managed identities to return. If
      unspecified, at most 50 managed identities are returned. The maximum
      value is 1000; values above are 1000 truncated to 1000.
    pageToken: A page token, received from a previous
      `ListWorkloadIdentityPoolManagedIdentities` call. Provide this to
      retrieve the subsequent page.
    parent: Required. The parent resource to list managed identities for.
    showDeleted: Whether to return soft-deleted managed identities.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)