from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesMetadata(_messages.Message):
    """KubernetesMetadata provides informational metadata for Memberships
  representing Kubernetes clusters.

  Fields:
    kubernetesApiServerVersion: Output only. Kubernetes API server version
      string as reported by `/version`.
    memoryMb: Output only. The total memory capacity as reported by the sum of
      all Kubernetes nodes resources, defined in MB.
    nodeCount: Output only. Node count as reported by Kubernetes nodes
      resources.
    nodeProviderId: Output only. Node providerID as reported by the first node
      in the list of nodes on the Kubernetes endpoint. On Kubernetes platforms
      that support zero-node clusters (like GKE-on-GCP), the node_count will
      be zero and the node_provider_id will be empty.
    updateTime: Output only. The time at which these details were last
      updated. This update_time is different from the Membership-level
      update_time since EndpointDetails are updated internally for API
      consumers.
    vcpuCount: Output only. vCPU count as reported by Kubernetes nodes
      resources.
  """
    kubernetesApiServerVersion = _messages.StringField(1)
    memoryMb = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    nodeCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    nodeProviderId = _messages.StringField(4)
    updateTime = _messages.StringField(5)
    vcpuCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)