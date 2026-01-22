from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApiMethodValueValuesEnum(_messages.Enum):
    """The OS policy assignment API method.

    Values:
      API_METHOD_UNSPECIFIED: Invalid value
      CREATE: Create OS policy assignment API method
      UPDATE: Update OS policy assignment API method
      DELETE: Delete OS policy assignment API method
    """
    API_METHOD_UNSPECIFIED = 0
    CREATE = 1
    UPDATE = 2
    DELETE = 3