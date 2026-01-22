from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiscoveredWorkload(_messages.Message):
    """DiscoveredWorkload is a binary deployment (such as managed instance
  groups (MIGs) and GKE deployments) that performs the smallest logical subset
  of business functionality. A discovered workload can be registered to an App
  Hub Workload.

  Fields:
    name: Identifier. The resource name of the discovered workload. Format:
      "projects/{host-project-
      id}/locations/{location}/discoveredWorkloads/{uuid}"
    workloadProperties: Output only. Properties of an underlying compute
      resource represented by the Workload. These are immutable.
    workloadReference: Output only. Reference of an underlying compute
      resource represented by the Workload. These are immutable.
  """
    name = _messages.StringField(1)
    workloadProperties = _messages.MessageField('WorkloadProperties', 2)
    workloadReference = _messages.MessageField('WorkloadReference', 3)