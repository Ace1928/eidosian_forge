from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrgMembership(_messages.Message):
    """A membership in an OrgUnit. An `OrgMembership` defines a relationship
  between an `OrgUnit` and an entity belonging to that `OrgUnit`, referred to
  as a "member".

  Enums:
    TypeValueValuesEnum: Immutable. Entity type for the org member.

  Fields:
    member: Immutable. Org member id as full resource name. Format for shared
      drive resource: //drive.googleapis.com/drives/{$memberId} where
      `$memberId` is the `id` from [Drive API (V3) `Drive` resource](https://d
      evelopers.google.com/drive/api/v3/reference/drives#resource).
    memberUri: Uri with which you can read the member. This follows
      https://aip.dev/122 Format for shared drive resource:
      https://drive.googleapis.com/drive/v3/drives/{$memberId} where
      `$memberId` is the `id` from [Drive API (V3) `Drive` resource](https://d
      evelopers.google.com/drive/api/v3/reference/drives#resource).
    name: Required. Immutable. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      OrgMembership. Format: orgUnits/{$orgUnitId}/memberships/{$membership}
      The `$orgUnitId` is the `orgUnitId` from the [Admin SDK `OrgUnit`
      resource](https://developers.google.com/admin-
      sdk/directory/reference/rest/v1/orgunits). The `$membership` shall be of
      the form `{$entityType};{$memberId}`, where `$entityType` is the enum
      value of [OrgMembership.EntityType], and `memberId` is the `id` from
      [Drive API (V3) `Drive` resource](https://developers.google.com/drive/ap
      i/v3/reference/drives#resource) for
      OrgMembership.EntityType.SHARED_DRIVE.
    type: Immutable. Entity type for the org member.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Immutable. Entity type for the org member.

    Values:
      ENTITY_TYPE_UNSPECIFIED: Equivalent to no resource type mentioned
      SHARED_DRIVE: Shared drive as resource type
    """
        ENTITY_TYPE_UNSPECIFIED = 0
        SHARED_DRIVE = 1
    member = _messages.StringField(1)
    memberUri = _messages.StringField(2)
    name = _messages.StringField(3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)