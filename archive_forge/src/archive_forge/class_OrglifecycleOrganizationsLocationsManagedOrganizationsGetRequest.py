from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrglifecycleOrganizationsLocationsManagedOrganizationsGetRequest(_messages.Message):
    """A OrglifecycleOrganizationsLocationsManagedOrganizationsGetRequest
  object.

  Fields:
    name: Required. The name of the ManagedOrganization to retrieve. Format: o
      rganizations/{organization_id}/locations/*/managedOrganizations/{managed
      _organization_id}
  """
    name = _messages.StringField(1, required=True)