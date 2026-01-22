from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceValueValueValuesEnum(_messages.Enum):
    """How valuable this resource is.

    Values:
      RESOURCE_VALUE_UNSPECIFIED: The resource value isn't specified.
      RESOURCE_VALUE_LOW: This is a low-value resource.
      RESOURCE_VALUE_MEDIUM: This is a medium-value resource.
      RESOURCE_VALUE_HIGH: This is a high-value resource.
    """
    RESOURCE_VALUE_UNSPECIFIED = 0
    RESOURCE_VALUE_LOW = 1
    RESOURCE_VALUE_MEDIUM = 2
    RESOURCE_VALUE_HIGH = 3