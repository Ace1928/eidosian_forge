from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActionIsolationValueValuesEnum(_messages.Enum):
    """Defines the isolation policy for actions on this instance. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      ACTION_ISOLATION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to OFF.
      ACTION_ISOLATION_OFF: Disables enforcing feature policies that guarantee
        action isolation.
      ACTION_ISOLATION_ENFORCED: Enforces setting feature policies that
        ensures actions within the RBE Instance are isolated from each other
        in a way deemed sufficient by ISE reviewers.
    """
    ACTION_ISOLATION_UNSPECIFIED = 0
    ACTION_ISOLATION_OFF = 1
    ACTION_ISOLATION_ENFORCED = 2