from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MethodTypesValueListEntryValuesEnum(_messages.Enum):
    """MethodTypesValueListEntryValuesEnum enum type.

    Values:
      METHOD_TYPE_UNSPECIFIED: Unspecified. Results in an error.
      CREATE: Constraint applied when creating the resource.
      UPDATE: Constraint applied when updating the resource.
      DELETE: Constraint applied when deleting the resource. Not supported
        yet.
    """
    METHOD_TYPE_UNSPECIFIED = 0
    CREATE = 1
    UPDATE = 2
    DELETE = 3