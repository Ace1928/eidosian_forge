from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CleanupPolicy(_messages.Message):
    """Artifact policy configuration for repository cleanup policies.

  Enums:
    ActionValueValuesEnum: Policy action.

  Fields:
    action: Policy action.
    condition: Policy condition for matching versions.
    id: The user-provided ID of the cleanup policy.
    mostRecentVersions: Policy condition for retaining a minimum number of
      versions. May only be specified with a Keep action.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Policy action.

    Values:
      ACTION_UNSPECIFIED: Action not specified.
      DELETE: Delete action.
      KEEP: Keep action.
    """
        ACTION_UNSPECIFIED = 0
        DELETE = 1
        KEEP = 2
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    condition = _messages.MessageField('CleanupPolicyCondition', 2)
    id = _messages.StringField(3)
    mostRecentVersions = _messages.MessageField('CleanupPolicyMostRecentVersions', 4)