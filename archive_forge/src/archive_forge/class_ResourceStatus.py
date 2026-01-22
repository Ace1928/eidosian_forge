from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceStatus(_messages.Message):
    """Status of a deployment resource.

  Enums:
    ResourceTypeValueValuesEnum: Output only. Resource type.
    StatusValueValuesEnum: Output only. Status of the resource.

  Fields:
    group: Group to which the resource belongs to.
    kind: Kind of the resource.
    name: Name of the resource.
    nfDeployStatus: Output only. Detailed status of NFDeploy.
    resourceNamespace: Namespace of the resource.
    resourceType: Output only. Resource type.
    status: Output only. Status of the resource.
    version: Version of the resource.
  """

    class ResourceTypeValueValuesEnum(_messages.Enum):
        """Output only. Resource type.

    Values:
      RESOURCE_TYPE_UNSPECIFIED: Unspecified resource type.
      NF_DEPLOY_RESOURCE: User specified NF Deploy CR.
      DEPLOYMENT_RESOURCE: CRs that are part of a blueprint.
    """
        RESOURCE_TYPE_UNSPECIFIED = 0
        NF_DEPLOY_RESOURCE = 1
        DEPLOYMENT_RESOURCE = 2

    class StatusValueValuesEnum(_messages.Enum):
        """Output only. Status of the resource.

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
    group = _messages.StringField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3)
    nfDeployStatus = _messages.MessageField('NFDeployStatus', 4)
    resourceNamespace = _messages.StringField(5)
    resourceType = _messages.EnumField('ResourceTypeValueValuesEnum', 6)
    status = _messages.EnumField('StatusValueValuesEnum', 7)
    version = _messages.StringField(8)