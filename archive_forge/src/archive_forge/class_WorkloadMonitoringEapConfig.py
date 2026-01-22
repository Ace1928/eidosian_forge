from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadMonitoringEapConfig(_messages.Message):
    """WorkloadMonitoringConfig is configuration for collecting workload
  metrics on GKE. Temporary config for EAP.

  Fields:
    enabled: Whether to send workload metrics from the cluster to Google Cloud
      Monitoring.
  """
    enabled = _messages.BooleanField(1)