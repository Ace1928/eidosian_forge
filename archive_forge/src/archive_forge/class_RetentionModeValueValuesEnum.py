from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetentionModeValueValuesEnum(_messages.Enum):
    """Optional. Retention can be either enabled or disabled.

    Values:
      RETENTION_MODE_UNSPECIFIED: Default mode doesn't change environment
        parameters.
      RETENTION_MODE_ENABLED: Retention policy is enabled.
      RETENTION_MODE_DISABLED: Retention policy is disabled.
    """
    RETENTION_MODE_UNSPECIFIED = 0
    RETENTION_MODE_ENABLED = 1
    RETENTION_MODE_DISABLED = 2