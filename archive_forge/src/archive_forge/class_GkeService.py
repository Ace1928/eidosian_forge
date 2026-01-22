from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeService(_messages.Message):
    """GKE Service. The "service" here represents a Kubernetes service object
  (https://kubernetes.io/docs/concepts/services-networking/service). The field
  names correspond to the resource labels on k8s_service monitored resources
  (https://cloud.google.com/monitoring/api/resources#tag_k8s_service).

  Fields:
    clusterName: The name of the parent cluster.
    location: The location of the parent cluster. This may be a zone or
      region.
    namespaceName: The name of the parent namespace.
    projectId: Output only. The project this resource lives in. For legacy
      services migrated from the Custom type, this may be a distinct project
      from the one parenting the service itself.
    serviceName: The name of this service.
  """
    clusterName = _messages.StringField(1)
    location = _messages.StringField(2)
    namespaceName = _messages.StringField(3)
    projectId = _messages.StringField(4)
    serviceName = _messages.StringField(5)