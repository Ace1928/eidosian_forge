from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2ConstraintGoogleDefinedCustomConstraint(_messages.Message):
    """A Google defined custom constraint. This represents a subset of fields
  missing from Constraint proto that are required to describe CustomConstraint

  Enums:
    ActionTypeValueValuesEnum: Allow or deny type.
    MethodTypesValueListEntryValuesEnum:

  Fields:
    actionType: Allow or deny type.
    condition: Org policy condition/expression. For example:
      `resource.instanceName.matches("[production|test]_.*_(\\d)+")` or,
      `resource.management.auto_upgrade == true` The max length of the
      condition is 1000 characters.
    methodTypes: All the operations being applied for this constraint.
    resourceTypes: The resource instance type on which this policy applies.
      Format will be of the form : `/` Example: *
      `compute.googleapis.com/Instance`.
  """

    class ActionTypeValueValuesEnum(_messages.Enum):
        """Allow or deny type.

    Values:
      ACTION_TYPE_UNSPECIFIED: Unspecified. Results in an error.
      ALLOW: Allowed action type.
      DENY: Deny action type.
    """
        ACTION_TYPE_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2

    class MethodTypesValueListEntryValuesEnum(_messages.Enum):
        """MethodTypesValueListEntryValuesEnum enum type.

    Values:
      METHOD_TYPE_UNSPECIFIED: Unspecified. Results in an error.
      CREATE: Constraint applied when creating the resource.
      UPDATE: Constraint applied when updating the resource.
      DELETE: Constraint applied when deleting the resource. Not supported
        yet.
    """
        METHOD_TYPE_UNSPECIFIED = 0
        CREATE = 1
        UPDATE = 2
        DELETE = 3
    actionType = _messages.EnumField('ActionTypeValueValuesEnum', 1)
    condition = _messages.StringField(2)
    methodTypes = _messages.EnumField('MethodTypesValueListEntryValuesEnum', 3, repeated=True)
    resourceTypes = _messages.StringField(4, repeated=True)