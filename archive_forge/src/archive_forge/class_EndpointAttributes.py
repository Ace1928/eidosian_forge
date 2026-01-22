from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointAttributes(_messages.Message):
    """Attributes associated with endpoints.

  Enums:
    KubernetesResourceTypeValueValuesEnum: Optional. Kubernetes resource-type
      associated with this endpoint

  Fields:
    gcpFleetMembership: Optional. Membership URI (scheme-less URI) for
      resources registered to Google Cloud Fleet. Currently populated only for
      kubernetes resources. Sample URI: `//gkehub.googleapis.com/projects/my-
      project/locations/global/memberships/my-membership`
    kubernetesResourceType: Optional. Kubernetes resource-type associated with
      this endpoint
    managedRegistration: Output only. Indicates whether a GCP product or
      service manages this resource. When a resource is fully managed by
      another GCP product or system the information in Service Directory is
      read-only. The source of truth is the relevant GCP product or system
      which is managing the resource. The Service Directory resource will be
      updated or deleted as appropriate to reflect the state of the underlying
      `origin_resource`.
    originResource: Optional. Reference to the underlying resource that this
      endpoint represents. This should be the full name of the resource that
      this endpoint was created from.
    region: Optional. Region of the underlying resource, or "global" for
      global resources.
    zone: Optional. GCP zone of the underlying resource. Meant to be populated
      only for zonal resources, left unset for others.
  """

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
    gcpFleetMembership = _messages.StringField(1)
    kubernetesResourceType = _messages.EnumField('KubernetesResourceTypeValueValuesEnum', 2)
    managedRegistration = _messages.BooleanField(3)
    originResource = _messages.StringField(4)
    region = _messages.StringField(5)
    zone = _messages.StringField(6)