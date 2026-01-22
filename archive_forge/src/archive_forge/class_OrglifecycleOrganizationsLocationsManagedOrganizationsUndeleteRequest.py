from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrglifecycleOrganizationsLocationsManagedOrganizationsUndeleteRequest(_messages.Message):
    """A OrglifecycleOrganizationsLocationsManagedOrganizationsUndeleteRequest
  object.

  Fields:
    name: Required. The name of the ManagedOrganization to delete. Format: org
      anizations/{organization_id}/locations/*/managedOrganizations/{managed_o
      rganization_id}
    undeleteManagedOrganizationRequest: A UndeleteManagedOrganizationRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteManagedOrganizationRequest = _messages.MessageField('UndeleteManagedOrganizationRequest', 2)