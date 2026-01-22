from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllValuesValueValuesEnum(_messages.Enum):
    """The policy all_values state.

    Values:
      ALL_VALUES_UNSPECIFIED: Indicates that allowed_values or denied_values
        must be set.
      ALLOW: A policy with this set allows all values.
      DENY: A policy with this set denies all values.
    """
    ALL_VALUES_UNSPECIFIED = 0
    ALLOW = 1
    DENY = 2