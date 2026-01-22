from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadPolicyConfig(_messages.Message):
    """WorkloadPolicyConfig is the configuration of workload policy for
  autopilot clusters.

  Fields:
    allowNetAdmin: If true, workloads can use NET_ADMIN capability.
  """
    allowNetAdmin = _messages.BooleanField(1)