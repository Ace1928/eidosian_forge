from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActionHermeticityValueValuesEnum(_messages.Enum):
    """Defines the hermeticity policy for actions on this instance. DO NOT
    USE: Experimental / unlaunched feature.

    Values:
      ACTION_HERMETICITY_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to OFF.
      ACTION_HERMETICITY_OFF: Disables enforcing feature policies that
        guarantee action hermeticity.
      ACTION_HERMETICITY_ENFORCED: Enforces hermeticity of actions by
        requiring feature policies to be set that prevent actions from gaining
        network access. The enforcement mechanism has been reviewed by ISE.
      ACTION_HERMETICITY_BEST_EFFORT: Requires feature policies to be set that
        provide best effort hermeticity for actions. Best effort hermeticity
        means network access will be disabled and not trivial to bypass.
        However, a determined and malicious action may still find a way to
        gain network access.
    """
    ACTION_HERMETICITY_UNSPECIFIED = 0
    ACTION_HERMETICITY_OFF = 1
    ACTION_HERMETICITY_ENFORCED = 2
    ACTION_HERMETICITY_BEST_EFFORT = 3