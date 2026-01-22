from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GlobalPolicyEvaluationModeValueValuesEnum(_messages.Enum):
    """Optional. Controls the evaluation of a Google-maintained global
    admission policy for common system-level images. Images not covered by the
    global policy will be subject to the project admission policy. This
    setting has no effect when specified inside a global admission policy.

    Values:
      GLOBAL_POLICY_EVALUATION_MODE_UNSPECIFIED: Not specified: DISABLE is
        assumed.
      ENABLE: Enables system policy evaluation.
      DISABLE: Disables system policy evaluation.
    """
    GLOBAL_POLICY_EVALUATION_MODE_UNSPECIFIED = 0
    ENABLE = 1
    DISABLE = 2