from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComparatorValueValuesEnum(_messages.Enum):
    """Comparator to use for comparing the field value.

    Values:
      COMPARATOR_UNSPECIFIED: The default value.
      EQUALS: The field value must be equal to the specified value.
      NOT_EQUALS: The field value must not be equal to the specified value.
    """
    COMPARATOR_UNSPECIFIED = 0
    EQUALS = 1
    NOT_EQUALS = 2