from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1CustomConstraint(_messages.Message):
    """The definition of a custom constraint.

  Enums:
    ActionTypeValueValuesEnum: Allow or deny type.
    MethodTypesValueListEntryValuesEnum:

  Fields:
    actionType: Allow or deny type.
    condition: Organization Policy condition/expression. For example:
      `resource.instanceName.matches("[production|test]_.*_(\\d)+")'` or,
      `resource.management.auto_upgrade == true`
    description: Detailed information about this custom policy constraint.
    displayName: One line display name for the UI.
    methodTypes: All the operations being applied for this constraint.
    name: Name of the constraint. This is unique within the organization.
      Format of the name should be * `organizations/{organization_id}/customCo
      nstraints/{custom_constraint_id}` Example :
      "organizations/123/customConstraints/custom.createOnlyE2TypeVms"
    resourceTypes: The Resource Instance type on which this policy applies to.
      Format will be of the form : "/" Example: *
      `compute.googleapis.com/Instance`.
  """

    class ActionTypeValueValuesEnum(_messages.Enum):
        """Allow or deny type.

    Values:
      ACTION_TYPE_UNSPECIFIED: Unspecified. Will results in user error.
      ALLOW: Allowed action type.
      DENY: Deny action type.
    """
        ACTION_TYPE_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2

    class MethodTypesValueListEntryValuesEnum(_messages.Enum):
        """MethodTypesValueListEntryValuesEnum enum type.

    Values:
      METHOD_TYPE_UNSPECIFIED: Unspecified. Will results in user error.
      CREATE: Constraint applied when creating the resource.
      UPDATE: Constraint applied when updating the resource.
      DELETE: Constraint applied when deleting the resource.
    """
        METHOD_TYPE_UNSPECIFIED = 0
        CREATE = 1
        UPDATE = 2
        DELETE = 3
    actionType = _messages.EnumField('ActionTypeValueValuesEnum', 1)
    condition = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    methodTypes = _messages.EnumField('MethodTypesValueListEntryValuesEnum', 5, repeated=True)
    name = _messages.StringField(6)
    resourceTypes = _messages.StringField(7, repeated=True)