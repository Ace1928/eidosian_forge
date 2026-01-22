from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeMetadataValueValuesEnum(_messages.Enum):
    """NodeMetadata is the configuration for how to expose metadata to the
    workloads running on the node.

    Values:
      UNSPECIFIED: Not set.
      SECURE: Prevent workloads not in hostNetwork from accessing certain VM
        metadata, specifically kube-env, which contains Kubelet credentials,
        and the instance identity token. Metadata concealment is a temporary
        security solution available while the bootstrapping process for
        cluster nodes is being redesigned with significant security
        improvements. This feature is scheduled to be deprecated in the future
        and later removed.
      EXPOSE: Expose all VM metadata to pods.
      GKE_METADATA_SERVER: Run the GKE Metadata Server on this node. The GKE
        Metadata Server exposes a metadata API to workloads that is compatible
        with the V1 Compute Metadata APIs exposed by the Compute Engine and
        App Engine Metadata Servers. This feature can only be enabled if
        Workload Identity is enabled at the cluster level.
    """
    UNSPECIFIED = 0
    SECURE = 1
    EXPOSE = 2
    GKE_METADATA_SERVER = 3