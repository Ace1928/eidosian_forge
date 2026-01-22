from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MacExecutionValueValuesEnum(_messages.Enum):
    """Defines how Windows actions are allowed to execute. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      MAC_EXECUTION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to FORBIDDEN.
      MAC_EXECUTION_FORBIDDEN: Mac actions and worker pools are forbidden.
      MAC_EXECUTION_UNRESTRICTED: No restrictions on execution of Mac actions.
    """
    MAC_EXECUTION_UNSPECIFIED = 0
    MAC_EXECUTION_FORBIDDEN = 1
    MAC_EXECUTION_UNRESTRICTED = 2