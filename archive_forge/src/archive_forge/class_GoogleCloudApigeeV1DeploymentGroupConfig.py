from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeploymentGroupConfig(_messages.Message):
    """DeploymentGroupConfig represents a deployment group that should be
  present in a particular environment.

  Enums:
    DeploymentGroupTypeValueValuesEnum: Type of the deployment group, which
      will be either Standard or Extensible.

  Fields:
    deploymentGroupType: Type of the deployment group, which will be either
      Standard or Extensible.
    name: Name of the deployment group in the following format:
      `organizations/{org}/environments/{env}/deploymentGroups/{group}`.
    revisionId: Revision number which can be used by the runtime to detect if
      the deployment group has changed between two versions.
    uid: Unique ID. The ID will only change if the deployment group is deleted
      and recreated.
  """

    class DeploymentGroupTypeValueValuesEnum(_messages.Enum):
        """Type of the deployment group, which will be either Standard or
    Extensible.

    Values:
      DEPLOYMENT_GROUP_TYPE_UNSPECIFIED: Unspecified type
      STANDARD: Standard type
      EXTENSIBLE: Extensible Type
    """
        DEPLOYMENT_GROUP_TYPE_UNSPECIFIED = 0
        STANDARD = 1
        EXTENSIBLE = 2
    deploymentGroupType = _messages.EnumField('DeploymentGroupTypeValueValuesEnum', 1)
    name = _messages.StringField(2)
    revisionId = _messages.IntegerField(3)
    uid = _messages.StringField(4)