from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorsValueListEntryValuesEnum(_messages.Enum):
    """ErrorsValueListEntryValuesEnum enum type.

    Values:
      ERROR_UNSPECIFIED: Default type.
      CUSTOM_CONTAINER: The GmN is configured with custom container(s) and
        cannot be migrated.
    """
    ERROR_UNSPECIFIED = 0
    CUSTOM_CONTAINER = 1