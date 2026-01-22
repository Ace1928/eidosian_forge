from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringConfig(_messages.Message):
    """MonitoringConfig informs Fleet-based applications/services/UIs how the
  metrics for the underlying cluster is reported to cloud monitoring services.
  It can be set from empty to non-empty, but can't be mutated directly to
  prevent accidentally breaking the constinousty of metrics.

  Fields:
    cluster: Optional. Cluster name used to report metrics. For Anthos on
      VMWare/Baremetal/MultiCloud clusters, it would be in format
      {cluster_type}/{cluster_name}, e.g., "awsClusters/cluster_1".
    clusterHash: Optional. For GKE and Multicloud clusters, this is the UUID
      of the cluster resource. For VMWare and Baremetal clusters, this is the
      kube-system UID.
    kubernetesMetricsPrefix: Optional. Kubernetes system metrics, if
      available, are written to this prefix. This defaults to kubernetes.io
      for GKE, and kubernetes.io/anthos for Anthos eventually. Noted: Anthos
      MultiCloud will have kubernetes.io prefix today but will migration to be
      under kubernetes.io/anthos.
    location: Optional. Location used to report Metrics
    projectId: Optional. Project used to report Metrics
  """
    cluster = _messages.StringField(1)
    clusterHash = _messages.StringField(2)
    kubernetesMetricsPrefix = _messages.StringField(3)
    location = _messages.StringField(4)
    projectId = _messages.StringField(5)