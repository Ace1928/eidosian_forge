from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysListRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsProvidersKeysListRequest
  object.

  Fields:
    pageSize: The maximum number of keys to return. If unspecified, all keys
      are returned. The maximum value is 10; values above 10 are truncated to
      10.
    pageToken: A page token, received from a previous
      `ListWorkloadIdentityPoolProviderKeys` call. Provide this to retrieve
      the subsequent page.
    parent: Required. The parent provider resource to list encryption keys
      for.
    showDeleted: Whether to return soft deleted resources as well.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)