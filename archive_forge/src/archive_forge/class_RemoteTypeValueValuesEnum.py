from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoteTypeValueValuesEnum(_messages.Enum):
    """RemoteTypeValueValuesEnum enum type.

    Values:
      REMOTE_TYPE_UNSPECIFIED: <no description>
      MIRROR: <no description>
      CACHE_LAYER: <no description>
    """
    REMOTE_TYPE_UNSPECIFIED = 0
    MIRROR = 1
    CACHE_LAYER = 2