from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendsValueListEntryValuesEnum(_messages.Enum):
    """BackendsValueListEntryValuesEnum enum type.

    Values:
      MONITORING_BACKEND_UNSPECIFIED: Backend cannot be determined
      PROMETHEUS: Prometheus backend for monitoring
      CLOUD_MONITORING: Stackdriver/Cloud Monitoring backend for monitoring
    """
    MONITORING_BACKEND_UNSPECIFIED = 0
    PROMETHEUS = 1
    CLOUD_MONITORING = 2