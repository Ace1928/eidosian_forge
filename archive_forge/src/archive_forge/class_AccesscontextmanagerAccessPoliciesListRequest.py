from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesListRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesListRequest object.

  Fields:
    pageSize: Number of AccessPolicy instances to include in the list. Default
      100.
    pageToken: Next page token for the next batch of AccessPolicy instances.
      Defaults to the first page of results.
    parent: Required. Resource name for the container to list AccessPolicy
      instances from. Format: `organizations/{org_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)