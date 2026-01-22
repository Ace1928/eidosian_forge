from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesServicePerimetersListRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesServicePerimetersListRequest object.

  Fields:
    pageSize: Number of Service Perimeters to include in the list. Default
      100.
    pageToken: Next page token for the next batch of Service Perimeter
      instances. Defaults to the first page of results.
    parent: Required. Resource name for the access policy to list Service
      Perimeters from. Format: `accessPolicies/{policy_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)