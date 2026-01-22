from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1WorkloadPartnerPermissions(_messages.Message):
    """Permissions granted to the AW Partner SA account for the customer
  workload

  Fields:
    assuredWorkloadsMonitoring: Optional. Allow partner to view violation
      alerts.
    dataLogsViewer: Allow the partner to view inspectability logs and
      monitoring violations.
    serviceAccessApprover: Optional. Allow partner to view access approval
      logs.
  """
    assuredWorkloadsMonitoring = _messages.BooleanField(1)
    dataLogsViewer = _messages.BooleanField(2)
    serviceAccessApprover = _messages.BooleanField(3)