from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlatformValueValuesEnum(_messages.Enum):
    """Platform hosting this deployment.

    Values:
      PLATFORM_UNSPECIFIED: Unknown.
      GKE: Google Container Engine.
      FLEX: Google App Engine: Flexible Environment.
      CUSTOM: Custom user-defined platform.
    """
    PLATFORM_UNSPECIFIED = 0
    GKE = 1
    FLEX = 2
    CUSTOM = 3