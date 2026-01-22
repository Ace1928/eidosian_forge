from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchDedicatedResources(_messages.Message):
    """A description of resources that are used for performing batch
  operations, are dedicated to a Model, and need manual configuration.

  Fields:
    machineSpec: Required. Immutable. The specification of a single machine.
    maxReplicaCount: Immutable. The maximum number of machine replicas the
      batch operation may be scaled to. The default value is 10.
    startingReplicaCount: Immutable. The number of machine replicas used at
      the start of the batch operation. If not set, Vertex AI decides starting
      number, not greater than max_replica_count
  """
    machineSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1MachineSpec', 1)
    maxReplicaCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    startingReplicaCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)