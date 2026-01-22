from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CanaryOptionValueValuesEnum(_messages.Enum):
    """The canary option set by the user upon setting breakpoint.

    Values:
      CANARY_OPTION_UNSPECIFIED: Depends on the canary_mode of the debuggee.
      CANARY_OPTION_TRY_ENABLE: Enable the canary for this breakpoint if the
        canary_mode of the debuggee is not CANARY_MODE_ALWAYS_ENABLED or
        CANARY_MODE_ALWAYS_DISABLED.
      CANARY_OPTION_TRY_DISABLE: Disable the canary for this breakpoint if the
        canary_mode of the debuggee is not CANARY_MODE_ALWAYS_ENABLED or
        CANARY_MODE_ALWAYS_DISABLED.
    """
    CANARY_OPTION_UNSPECIFIED = 0
    CANARY_OPTION_TRY_ENABLE = 1
    CANARY_OPTION_TRY_DISABLE = 2