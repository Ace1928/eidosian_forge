from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterOptions(_messages.Message):
    """Details of the GKE Cluster for builds that should execute on-cluster.

  Fields:
    name: Identifier of the GKE Cluster this build should execute on. Example:
      projects/{project_id}/locations/{location}/clusters/{cluster_name} The
      cluster's project ID must be the same project ID that is running the
      build. The cluster must exist and have the CloudBuild add-on enabled.
  """
    name = _messages.StringField(1)