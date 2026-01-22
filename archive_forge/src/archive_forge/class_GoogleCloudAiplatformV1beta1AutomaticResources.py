from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AutomaticResources(_messages.Message):
    """A description of resources that to large degree are decided by Vertex
  AI, and require only a modest additional configuration. Each Model
  supporting these resources documents its specific guidelines.

  Fields:
    maxReplicaCount: Immutable. The maximum number of replicas this
      DeployedModel may be deployed on when the traffic against it increases.
      If the requested value is too large, the deployment will error, but if
      deployment succeeds then the ability to scale the model to that many
      replicas is guaranteed (barring service outages). If traffic against the
      DeployedModel increases beyond what its replicas at maximum may handle,
      a portion of the traffic will be dropped. If this value is not provided,
      a no upper bound for scaling under heavy traffic will be assume, though
      Vertex AI may be unable to scale beyond certain replica number.
    minReplicaCount: Immutable. The minimum number of replicas this
      DeployedModel will be always deployed on. If traffic against it
      increases, it may dynamically be deployed onto more replicas up to
      max_replica_count, and as traffic decreases, some of these extra
      replicas may be freed. If the requested value is too large, the
      deployment will error.
  """
    maxReplicaCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minReplicaCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)