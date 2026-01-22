from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespaceLifecycleState(_messages.Message):
    """NamespaceLifecycleState describes the state of a Namespace resource.

  Enums:
    CodeValueValuesEnum: Output only. The current state of the Namespace
      resource.

  Fields:
    code: Output only. The current state of the Namespace resource.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. The current state of the Namespace resource.

    Values:
      CODE_UNSPECIFIED: The code is not set.
      CREATING: The namespace is being created.
      READY: The namespace active.
      DELETING: The namespace is being deleted.
      UPDATING: The namespace is being updated.
    """
        CODE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3
        UPDATING = 4
    code = _messages.EnumField('CodeValueValuesEnum', 1)