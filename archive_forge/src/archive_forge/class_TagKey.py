from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagKey(_messages.Message):
    """A TagKey, used to group a set of TagValues.

  Enums:
    PurposeValueValuesEnum: Optional. A purpose denotes that this Tag is
      intended for use in policies of a specific policy engine, and will
      involve that policy engine in management operations involving this Tag.
      A purpose does not grant a policy engine exclusive rights to the Tag,
      and it may be referenced by other policy engines. A purpose cannot be
      changed once set.

  Messages:
    PurposeDataValue: Optional. Purpose data corresponds to the policy system
      that the tag is intended for. See documentation for `Purpose` for
      formatting of this field. Purpose data cannot be changed once set.

  Fields:
    createTime: Output only. Creation time.
    description: Optional. User-assigned description of the TagKey. Must not
      exceed 256 characters. Read-write.
    etag: Optional. Entity tag which users can pass to prevent race
      conditions. This field is always set in server responses. See
      UpdateTagKeyRequest for details.
    name: Immutable. The resource name for a TagKey. Must be in the format
      `tagKeys/{tag_key_id}`, where `tag_key_id` is the generated numeric id
      for the TagKey.
    namespacedName: Output only. Immutable. Namespaced name of the TagKey.
    parent: Immutable. The resource name of the TagKey's parent. A TagKey can
      be parented by an Organization or a Project. For a TagKey parented by an
      Organization, its parent must be in the form `organizations/{org_id}`.
      For a TagKey parented by a Project, its parent can be in the form
      `projects/{project_id}` or `projects/{project_number}`.
    purpose: Optional. A purpose denotes that this Tag is intended for use in
      policies of a specific policy engine, and will involve that policy
      engine in management operations involving this Tag. A purpose does not
      grant a policy engine exclusive rights to the Tag, and it may be
      referenced by other policy engines. A purpose cannot be changed once
      set.
    purposeData: Optional. Purpose data corresponds to the policy system that
      the tag is intended for. See documentation for `Purpose` for formatting
      of this field. Purpose data cannot be changed once set.
    shortName: Required. Immutable. The user friendly name for a TagKey. The
      short name should be unique for TagKeys within the same tag namespace.
      The short name must be 1-63 characters, beginning and ending with an
      alphanumeric character ([a-z0-9A-Z]) with dashes (-), underscores (_),
      dots (.), and alphanumerics between.
    updateTime: Output only. Update time.
  """

    class PurposeValueValuesEnum(_messages.Enum):
        """Optional. A purpose denotes that this Tag is intended for use in
    policies of a specific policy engine, and will involve that policy engine
    in management operations involving this Tag. A purpose does not grant a
    policy engine exclusive rights to the Tag, and it may be referenced by
    other policy engines. A purpose cannot be changed once set.

    Values:
      PURPOSE_UNSPECIFIED: Unspecified purpose.
      GCE_FIREWALL: Purpose for Compute Engine firewalls. A corresponding
        `purpose_data` should be set for the network the tag is intended for.
        The key should be `network` and the value should be in ## either of
        these two formats: `https://www.googleapis.com/compute/{compute_versio
        n}/projects/{project_id}/global/networks/{network_id}` -
        `{project_id}/{network_name}` ## Examples:
        `https://www.googleapis.com/compute/staging_v1/projects/fail-closed-
        load-testing/global/networks/6992953698831725600` - `fail-closed-load-
        testing/load-testing-network`
      DATA_GOVERNANCE: Purpose for data governance. Tag Values created under a
        key with this purpose may have Tag Value children. No `purpose_data`
        should be set.
    """
        PURPOSE_UNSPECIFIED = 0
        GCE_FIREWALL = 1
        DATA_GOVERNANCE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PurposeDataValue(_messages.Message):
        """Optional. Purpose data corresponds to the policy system that the tag
    is intended for. See documentation for `Purpose` for formatting of this
    field. Purpose data cannot be changed once set.

    Messages:
      AdditionalProperty: An additional property for a PurposeDataValue
        object.

    Fields:
      additionalProperties: Additional properties of type PurposeDataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PurposeDataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    namespacedName = _messages.StringField(5)
    parent = _messages.StringField(6)
    purpose = _messages.EnumField('PurposeValueValuesEnum', 7)
    purposeData = _messages.MessageField('PurposeDataValue', 8)
    shortName = _messages.StringField(9)
    updateTime = _messages.StringField(10)