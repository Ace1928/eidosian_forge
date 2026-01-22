from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class KeyTypesValueValuesEnum(_messages.Enum):
    """Filters the types of keys the user wants to include in the list
    response. Duplicate key types are not allowed. If no key type is provided,
    all keys are returned.

    Values:
      KEY_TYPE_UNSPECIFIED: <no description>
      USER_MANAGED: <no description>
      SYSTEM_MANAGED: <no description>
    """
    KEY_TYPE_UNSPECIFIED = 0
    USER_MANAGED = 1
    SYSTEM_MANAGED = 2