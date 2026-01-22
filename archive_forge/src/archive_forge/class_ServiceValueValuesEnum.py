from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceValueValuesEnum(_messages.Enum):
    """Service that physically stores the data.

    Values:
      SERVICE_UNSPECIFIED: Default unknown service.
      CLOUD_STORAGE: Google Cloud Storage service.
      BIGQUERY: BigQuery service.
    """
    SERVICE_UNSPECIFIED = 0
    CLOUD_STORAGE = 1
    BIGQUERY = 2