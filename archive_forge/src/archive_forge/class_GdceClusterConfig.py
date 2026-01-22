from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GdceClusterConfig(_messages.Message):
    """The target GDCE cluster config.

  Fields:
    gdcEdgeIdentityProvider: Optional. The name of the identity provider
      associated with the GDCE cluster.
    gdcEdgeMembershipTarget: Optional. A target GDCE cluster to deploy to. It
      must be in the same project and region as the Dataproc cluster'. Format:
      'projects/{project}/locations/{location}/clusters/{cluster_id}'
    gdcEdgeWorkloadIdentityPool: Optional. The workload identity pool
      associated with the fleet.
  """
    gdcEdgeIdentityProvider = _messages.StringField(1)
    gdcEdgeMembershipTarget = _messages.StringField(2)
    gdcEdgeWorkloadIdentityPool = _messages.StringField(3)