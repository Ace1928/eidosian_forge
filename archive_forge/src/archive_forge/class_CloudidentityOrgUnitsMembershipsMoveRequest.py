from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityOrgUnitsMembershipsMoveRequest(_messages.Message):
    """A CloudidentityOrgUnitsMembershipsMoveRequest object.

  Fields:
    moveOrgMembershipRequest: A MoveOrgMembershipRequest resource to be passed
      as the request body.
    name: Required. Immutable. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      OrgMembership. Format: orgUnits/{$orgUnitId}/memberships/{$membership}
      The `$orgUnitId` is the `orgUnitId` from the [Admin SDK `OrgUnit`
      resource](https://developers.google.com/admin-
      sdk/directory/reference/rest/v1/orgunits). To manage a Membership
      without specifying source `orgUnitId`, this API also supports the
      wildcard character '-' for `$orgUnitId` per https://google.aip.dev/159.
      The `$membership` shall be of the form `{$entityType};{$memberId}`,
      where `$entityType` is the enum value of OrgMembership.EntityType, and
      `memberId` is the `id` from [Drive API (V3) `Drive` resource](https://de
      velopers.google.com/drive/api/v3/reference/drives#resource) for
      OrgMembership.EntityType.SHARED_DRIVE.
  """
    moveOrgMembershipRequest = _messages.MessageField('MoveOrgMembershipRequest', 1)
    name = _messages.StringField(2, required=True)