from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScopeLifecycleState(_messages.Message):
    """ScopeLifecycleState describes the state of a Scope resource.

  Enums:
    CodeValueValuesEnum: Output only. The current state of the scope resource.

  Fields:
    code: Output only. The current state of the scope resource.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. The current state of the scope resource.

    Values:
      CODE_UNSPECIFIED: The code is not set.
      CREATING: The scope is being created.
      READY: The scope active.
      DELETING: The scope is being deleted.
      UPDATING: The scope is being updated.
    """
        CODE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3
        UPDATING = 4
    code = _messages.EnumField('CodeValueValuesEnum', 1)