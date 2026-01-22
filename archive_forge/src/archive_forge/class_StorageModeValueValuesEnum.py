from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageModeValueValuesEnum(_messages.Enum):
    """Optional. The mode of storage for Airflow workers task logs.

    Values:
      TASK_LOGS_STORAGE_MODE_UNSPECIFIED: This configuration is not specified
        by the user.
      CLOUD_LOGGING_AND_CLOUD_STORAGE: Store task logs in Cloud Logging and in
        the environment's Cloud Storage bucket.
      CLOUD_LOGGING_ONLY: Store task logs in Cloud Logging only.
    """
    TASK_LOGS_STORAGE_MODE_UNSPECIFIED = 0
    CLOUD_LOGGING_AND_CLOUD_STORAGE = 1
    CLOUD_LOGGING_ONLY = 2