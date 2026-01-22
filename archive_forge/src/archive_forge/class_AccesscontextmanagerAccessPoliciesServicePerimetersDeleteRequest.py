from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesServicePerimetersDeleteRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesServicePerimetersDeleteRequest
  object.

  Fields:
    name: Required. Resource name for the Service Perimeter. Format:
      `accessPolicies/{policy_id}/servicePerimeters/{service_perimeter_id}`
  """
    name = _messages.StringField(1, required=True)