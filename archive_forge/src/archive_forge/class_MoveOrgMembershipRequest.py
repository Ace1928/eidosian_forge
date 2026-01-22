from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveOrgMembershipRequest(_messages.Message):
    """The request message for OrgMembershipsService.MoveOrgMembership.

  Fields:
    customer: Required. Immutable. Customer on whose membership change is
      made. All authorization will happen on the role assignments of this
      customer. Format: customers/{$customerId} where `$customerId` is the
      `id` from the [Admin SDK `Customer`
      resource](https://developers.google.com/admin-
      sdk/directory/reference/rest/v1/customers). You may also use
      `customers/my_customer` to specify your own organization.
    destinationOrgUnit: Required. Immutable. OrgUnit where the membership will
      be moved to. Format: orgUnits/{$orgUnitId} where `$orgUnitId` is the
      `orgUnitId` from the [Admin SDK `OrgUnit`
      resource](https://developers.google.com/admin-
      sdk/directory/reference/rest/v1/orgunits).
  """
    customer = _messages.StringField(1)
    destinationOrgUnit = _messages.StringField(2)