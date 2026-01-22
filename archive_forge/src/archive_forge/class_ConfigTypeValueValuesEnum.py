from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigTypeValueValuesEnum(_messages.Enum):
    """Output only. Whether this instance config is a Google or User Managed
    Configuration.

    Values:
      TYPE_UNSPECIFIED: Unspecified.
      GOOGLE_MANAGED: Google managed configuration.
      USER_MANAGED: User managed configuration.
    """
    TYPE_UNSPECIFIED = 0
    GOOGLE_MANAGED = 1
    USER_MANAGED = 2