from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BehaviorsValueListEntryValuesEnum(_messages.Enum):
    """BehaviorsValueListEntryValuesEnum enum type.

    Values:
      DATA_SAMPLING_BEHAVIOR_UNSPECIFIED: If given, has no effect on sampling
        behavior. Used as an unknown or unset sentinel value.
      DISABLED: When given, disables element sampling. Has same behavior as
        not setting the behavior.
      ALWAYS_ON: When given, enables sampling in-flight from all PCollections.
      EXCEPTIONS: When given, enables sampling input elements when a user-
        defined DoFn causes an exception.
    """
    DATA_SAMPLING_BEHAVIOR_UNSPECIFIED = 0
    DISABLED = 1
    ALWAYS_ON = 2
    EXCEPTIONS = 3