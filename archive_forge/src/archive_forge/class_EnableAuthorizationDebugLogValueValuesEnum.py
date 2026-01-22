from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableAuthorizationDebugLogValueValuesEnum(_messages.Enum):
    """Optional. Enable the generation of authorization debug logs for the
    target.

    Values:
      LOG_NONE: Disable the authorization debug log.
      LOG_ERROR: Generate authorization debug log only in case the
        authorization result is an error.
      LOG_DENY_AND_ERROR: Generate authorization debug log only in case the
        authorization is denied or the authorization result is an error.
      LOG_ALL: Generate authorization debug log for all the authorization
        results.
    """
    LOG_NONE = 0
    LOG_ERROR = 1
    LOG_DENY_AND_ERROR = 2
    LOG_ALL = 3