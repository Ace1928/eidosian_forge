from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntentValueValuesEnum(_messages.Enum):
    """Output only. Intent of the resource.

    Values:
      INTENT_UNSPECIFIED: The default value. This value is used if the intent
        is omitted.
      CREATE: Infra Manager will create this Resource.
      UPDATE: Infra Manager will update this Resource.
      DELETE: Infra Manager will delete this Resource.
      RECREATE: Infra Manager will destroy and recreate this Resource.
      UNCHANGED: Infra Manager will leave this Resource untouched.
    """
    INTENT_UNSPECIFIED = 0
    CREATE = 1
    UPDATE = 2
    DELETE = 3
    RECREATE = 4
    UNCHANGED = 5