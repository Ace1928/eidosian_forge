from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogNameValueValuesEnum(_messages.Enum):
    """The log_name to populate in the Cloud Audit Record.

    Values:
      UNSPECIFIED_LOG_NAME: Default. Should not be used.
      ADMIN_ACTIVITY: Corresponds to "cloudaudit.googleapis.com/activity"
      DATA_ACCESS: Corresponds to "cloudaudit.googleapis.com/data_access"
    """
    UNSPECIFIED_LOG_NAME = 0
    ADMIN_ACTIVITY = 1
    DATA_ACCESS = 2