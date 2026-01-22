from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadClusterTypeValueValuesEnum(_messages.Enum):
    """Optional. Type of workload cluster for which an EdgeSLM resource is
    created.

    Values:
      WORKLOAD_CLUSTER_TYPE_UNSPECIFIED: Unspecified workload cluster.
      GDCE: Workload cluster is a GDCE cluster.
      GKE: Workload cluster is a GKE cluster.
    """
    WORKLOAD_CLUSTER_TYPE_UNSPECIFIED = 0
    GDCE = 1
    GKE = 2