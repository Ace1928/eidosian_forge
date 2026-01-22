from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CanaryModeValueValuesEnum(_messages.Enum):
    """Used when setting breakpoint canary for this debuggee.

    Values:
      CANARY_MODE_UNSPECIFIED: CANARY_MODE_UNSPECIFIED is equivalent to
        CANARY_MODE_ALWAYS_DISABLED so that if the debuggee is not configured
        to use the canary feature, the feature will be disabled.
      CANARY_MODE_ALWAYS_ENABLED: Always enable breakpoint canary regardless
        of the value of breakpoint's canary option.
      CANARY_MODE_ALWAYS_DISABLED: Always disable breakpoint canary regardless
        of the value of breakpoint's canary option.
      CANARY_MODE_DEFAULT_ENABLED: Depends on the breakpoint's canary option.
        Enable canary by default if the breakpoint's canary option is not
        specified.
      CANARY_MODE_DEFAULT_DISABLED: Depends on the breakpoint's canary option.
        Disable canary by default if the breakpoint's canary option is not
        specified.
    """
    CANARY_MODE_UNSPECIFIED = 0
    CANARY_MODE_ALWAYS_ENABLED = 1
    CANARY_MODE_ALWAYS_DISABLED = 2
    CANARY_MODE_DEFAULT_ENABLED = 3
    CANARY_MODE_DEFAULT_DISABLED = 4