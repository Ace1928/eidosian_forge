from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BindingDelta(_messages.Message):
    """One delta entry for Binding. Each individual change (only one member in
  each entry) to a binding will be a separate entry.

  Enums:
    ActionValueValuesEnum: The action that was performed on a Binding.
      Required

  Fields:
    action: The action that was performed on a Binding. Required
    condition: The condition that is associated with this binding.
    member: A single identity requesting access for a Google Cloud resource.
      Follows the same format of Binding.members. Required
    role: Role that is assigned to `members`. For example, `roles/viewer`,
      `roles/editor`, or `roles/owner`. Required
  """

    class ActionValueValuesEnum(_messages.Enum):
        """The action that was performed on a Binding. Required

    Values:
      ACTION_UNSPECIFIED: Unspecified.
      ADD: Addition of a Binding.
      REMOVE: Removal of a Binding.
    """
        ACTION_UNSPECIFIED = 0
        ADD = 1
        REMOVE = 2
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    condition = _messages.MessageField('Expr', 2)
    member = _messages.StringField(3)
    role = _messages.StringField(4)