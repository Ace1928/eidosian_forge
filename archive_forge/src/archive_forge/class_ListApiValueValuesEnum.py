from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListApiValueValuesEnum(_messages.Enum):
    """The Listing API to use for discovering objects. When not specified,
    Transfer Service will attempt to determine the right API to use.

    Values:
      LIST_API_UNSPECIFIED: ListApi is not specified.
      LIST_OBJECTS_V2: Perform listing using ListObjectsV2 API.
      LIST_OBJECTS: Legacy ListObjects API.
    """
    LIST_API_UNSPECIFIED = 0
    LIST_OBJECTS_V2 = 1
    LIST_OBJECTS = 2