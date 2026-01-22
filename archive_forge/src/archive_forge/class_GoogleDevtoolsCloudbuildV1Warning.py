from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1Warning(_messages.Message):
    """A non-fatal problem encountered during the execution of the build.

  Enums:
    PriorityValueValuesEnum: The priority for this warning.

  Fields:
    priority: The priority for this warning.
    text: Explanation of the warning generated.
  """

    class PriorityValueValuesEnum(_messages.Enum):
        """The priority for this warning.

    Values:
      PRIORITY_UNSPECIFIED: Should not be used.
      INFO: e.g. deprecation warnings and alternative feature highlights.
      WARNING: e.g. automated detection of possible issues with the build.
      ALERT: e.g. alerts that a feature used in the build is pending removal
    """
        PRIORITY_UNSPECIFIED = 0
        INFO = 1
        WARNING = 2
        ALERT = 3
    priority = _messages.EnumField('PriorityValueValuesEnum', 1)
    text = _messages.StringField(2)