from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BgpBestPathSelectionModeValueValuesEnum(_messages.Enum):
    """The BGP best path selection algorithm to be employed within this
    network for dynamic routes learned by Cloud Routers. Can be LEGACY
    (default) or STANDARD.

    Values:
      LEGACY: <no description>
      STANDARD: <no description>
    """
    LEGACY = 0
    STANDARD = 1