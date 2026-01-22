from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestartTypeValueValuesEnum(_messages.Enum):
    """Whether the Instance should be automatically restarted whenever it is
    terminated by Compute Engine (not terminated by user). This configuration
    is identical to `automaticRestart` field in Compute Engine create instance
    under scheduling. It was changed to an enum (instead of a boolean) to
    match the default value in Compute Engine which is automatic restart.

    Values:
      RESTART_TYPE_UNSPECIFIED: Unspecified behavior. This will use the
        default.
      AUTOMATIC_RESTART: The Instance should be automatically restarted
        whenever it is terminated by Compute Engine.
      NO_AUTOMATIC_RESTART: The Instance isn't automatically restarted
        whenever it is terminated by Compute Engine.
    """
    RESTART_TYPE_UNSPECIFIED = 0
    AUTOMATIC_RESTART = 1
    NO_AUTOMATIC_RESTART = 2