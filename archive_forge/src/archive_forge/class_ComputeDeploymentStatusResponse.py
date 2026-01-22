from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeDeploymentStatusResponse(_messages.Message):
    """Response object for `ComputeDeploymentStatus`.

  Enums:
    AggregatedStatusValueValuesEnum: Output only. Aggregated status of a
      deployment.

  Fields:
    aggregatedStatus: Output only. Aggregated status of a deployment.
    name: The name of the deployment.
    resourceStatuses: Output only. Resource level status details in
      deployments.
  """

    class AggregatedStatusValueValuesEnum(_messages.Enum):
        """Output only. Aggregated status of a deployment.

    Values:
      STATUS_UNSPECIFIED: Unknown state.
      STATUS_IN_PROGRESS: Under progress.
      STATUS_ACTIVE: Running and ready to serve traffic.
      STATUS_FAILED: Failed or stalled.
      STATUS_DELETING: Delete in progress.
      STATUS_DELETED: Deleted deployment.
      STATUS_PEERING: NFDeploy specific status. Peering in progress.
      STATUS_NOT_APPLICABLE: K8s objects such as NetworkAttachmentDefinition
        don't have a defined status.
    """
        STATUS_UNSPECIFIED = 0
        STATUS_IN_PROGRESS = 1
        STATUS_ACTIVE = 2
        STATUS_FAILED = 3
        STATUS_DELETING = 4
        STATUS_DELETED = 5
        STATUS_PEERING = 6
        STATUS_NOT_APPLICABLE = 7
    aggregatedStatus = _messages.EnumField('AggregatedStatusValueValuesEnum', 1)
    name = _messages.StringField(2)
    resourceStatuses = _messages.MessageField('ResourceStatus', 3, repeated=True)