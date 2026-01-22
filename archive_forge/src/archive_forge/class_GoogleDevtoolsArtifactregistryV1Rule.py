from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1Rule(_messages.Message):
    """Rules point to a version and represent an alternative name that can be
  used to access the version.

  Enums:
    ActionValueValuesEnum: What action this rule would make.
    OperationValueValuesEnum:

  Fields:
    action: What action this rule would make.
    condition: Optional. CEL expression. If not provided, the rule matches all
      the objects.
    name: The name of the rule, for example: "projects/p1/locations/us-
      central1/repositories/repo1/rules/rule1".
    operation: A OperationValueValuesEnum attribute.
    packageId: If empty, this rule is targeting all the packages inside the
      repository. If provided, the rule will only be applied to the package.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """What action this rule would make.

    Values:
      ACTION_UNSPECIFIED: Action not specified, treated as allow.
      ALLOW: Allow the operation.
      DENY: Deny the operation.
    """
        ACTION_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2

    class OperationValueValuesEnum(_messages.Enum):
        """OperationValueValuesEnum enum type.

    Values:
      OPERATION_UNSPECIFIED: Operation not specified.
      DOWNLOAD: Download operation.
    """
        OPERATION_UNSPECIFIED = 0
        DOWNLOAD = 1
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    condition = _messages.MessageField('Expr', 2)
    name = _messages.StringField(3)
    operation = _messages.EnumField('OperationValueValuesEnum', 4)
    packageId = _messages.StringField(5)