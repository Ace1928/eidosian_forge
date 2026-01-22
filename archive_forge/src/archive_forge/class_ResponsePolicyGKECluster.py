from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResponsePolicyGKECluster(_messages.Message):
    """A ResponsePolicyGKECluster object.

  Fields:
    gkeClusterName: The resource name of the cluster to bind this response
      policy to. This should be specified in the format like:
      projects/*/locations/*/clusters/*. This is referenced from GKE
      projects.locations.clusters.get API:
      https://cloud.google.com/kubernetes-
      engine/docs/reference/rest/v1/projects.locations.clusters/get
    kind: A string attribute.
  """
    gkeClusterName = _messages.StringField(1)
    kind = _messages.StringField(2, default='dns#responsePolicyGKECluster')