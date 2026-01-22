from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetMonitoringServiceRequest(_messages.Message):
    """SetMonitoringServiceRequest sets the monitoring service of a cluster.

  Fields:
    clusterId: Deprecated. The name of the cluster to upgrade. This field has
      been deprecated and replaced by the name field.
    monitoringService: Required. The monitoring service the cluster should use
      to write metrics. Currently available options: *
      "monitoring.googleapis.com/kubernetes" - The Cloud Monitoring service
      with a Kubernetes-native resource model * `monitoring.googleapis.com` -
      The legacy Cloud Monitoring service (no longer available as of GKE
      1.15). * `none` - No metrics will be exported from the cluster. If left
      as an empty string,`monitoring.googleapis.com/kubernetes` will be used
      for GKE 1.14+ or `monitoring.googleapis.com` for earlier versions.
    name: The name (project, location, cluster) of the cluster to set
      monitoring. Specified in the format `projects/*/locations/*/clusters/*`.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """
    clusterId = _messages.StringField(1)
    monitoringService = _messages.StringField(2)
    name = _messages.StringField(3)
    projectId = _messages.StringField(4)
    zone = _messages.StringField(5)