from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OptionalValueValuesEnum(_messages.Enum):
    """Deprecated in favor of optionalMode. This field can only be specified
    if logging is enabled for this backend service. Configures whether all,
    none or a subset of optional fields should be added to the reported logs.
    One of [INCLUDE_ALL_OPTIONAL, EXCLUDE_ALL_OPTIONAL, CUSTOM]. Default is
    EXCLUDE_ALL_OPTIONAL.

    Values:
      CUSTOM: A subset of optional fields.
      EXCLUDE_ALL_OPTIONAL: None optional fields.
      INCLUDE_ALL_OPTIONAL: All optional fields.
      UNSPECIFIED_OPTIONAL_MODE: <no description>
    """
    CUSTOM = 0
    EXCLUDE_ALL_OPTIONAL = 1
    INCLUDE_ALL_OPTIONAL = 2
    UNSPECIFIED_OPTIONAL_MODE = 3