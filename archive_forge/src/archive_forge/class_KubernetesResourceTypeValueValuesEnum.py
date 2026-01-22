from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesResourceTypeValueValuesEnum(_messages.Enum):
    """Optional. Kubernetes resource-type associated with this endpoint

    Values:
      KUBERNETES_RESOURCE_TYPE_UNSPECIFIED: Not a Kubernetes workload.
      KUBERNETES_RESOURCE_TYPE_CLUSTER_IP: Cluster IP service related resource
      KUBERNETES_RESOURCE_TYPE_NODE_PORT: Node port service related resource
      KUBERNETES_RESOURCE_TYPE_LOAD_BALANCER: Load balancer service related
        resource
      KUBERNETES_RESOURCE_TYPE_HEADLESS: Headless service related resource
    """
    KUBERNETES_RESOURCE_TYPE_UNSPECIFIED = 0
    KUBERNETES_RESOURCE_TYPE_CLUSTER_IP = 1
    KUBERNETES_RESOURCE_TYPE_NODE_PORT = 2
    KUBERNETES_RESOURCE_TYPE_LOAD_BALANCER = 3
    KUBERNETES_RESOURCE_TYPE_HEADLESS = 4