from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeWorkload(_messages.Message):
    """A GKE Workload (Deployment, StatefulSet, etc). The field names
  correspond to the metadata labels on monitored resources that fall under a
  workload (for example, k8s_container or k8s_pod).

  Fields:
    clusterName: The name of the parent cluster.
    location: The location of the parent cluster. This may be a zone or
      region.
    namespaceName: The name of the parent namespace.
    projectId: Output only. The project this resource lives in. For legacy
      services migrated from the Custom type, this may be a distinct project
      from the one parenting the service itself.
    topLevelControllerName: The name of this workload.
    topLevelControllerType: The type of this workload (for example,
      "Deployment" or "DaemonSet")
  """
    clusterName = _messages.StringField(1)
    location = _messages.StringField(2)
    namespaceName = _messages.StringField(3)
    projectId = _messages.StringField(4)
    topLevelControllerName = _messages.StringField(5)
    topLevelControllerType = _messages.StringField(6)