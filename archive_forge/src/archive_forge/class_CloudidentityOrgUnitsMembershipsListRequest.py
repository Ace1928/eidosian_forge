from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityOrgUnitsMembershipsListRequest(_messages.Message):
    """A CloudidentityOrgUnitsMembershipsListRequest object.

  Fields:
    customer: Required. Immutable. Customer that this OrgMembership belongs
      to. All authorization will happen on the role assignments of this
      customer. Format: customers/{$customerId} where `$customerId` is the
      `id` from the [Admin SDK `Customer`
      resource](https://developers.google.com/admin-
      sdk/directory/reference/rest/v1/customers). You may also use
      `customers/my_customer` to specify your own organization.
    filter: The search query. Must be specified in [Common Expression
      Language](https://opensource.google/projects/cel). May only contain
      equality operators on the `type` (e.g., `type == 'shared_drive'`).
    pageSize: The maximum number of results to return. The service may return
      fewer than this value. If omitted (or defaulted to zero) the server will
      default to 50. The maximum allowed value is 100, though requests with
      page_size greater than that will be silently interpreted as 100.
    pageToken: A page token, received from a previous
      `OrgMembershipsService.ListOrgMemberships` call. Provide this to
      retrieve the subsequent page. When paginating, all other parameters
      provided to `ListOrgMembershipsRequest` must match the call that
      provided the page token.
    parent: Required. Immutable. OrgUnit which is queried for a list of
      memberships. Format: orgUnits/{$orgUnitId} where `$orgUnitId` is the
      `orgUnitId` from the [Admin SDK `OrgUnit`
      resource](https://developers.google.com/admin-
      sdk/directory/reference/rest/v1/orgunits).
  """
    customer = _messages.StringField(1)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)