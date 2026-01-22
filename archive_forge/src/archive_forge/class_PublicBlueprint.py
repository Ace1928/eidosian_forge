from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicBlueprint(_messages.Message):
    """A Blueprint contains a collection of kubernetes resources in the form of
  YAML files. The file contents of a blueprint are collectively known as
  package. Public blueprint is a TNA provided blueprint that in present in
  TNA's public catalog. A user can copy the public blueprint to their private
  catalog for further modifications.

  Enums:
    DeploymentLevelValueValuesEnum: DeploymentLevel of a blueprint signifies
      where the blueprint will be applied. e.g. [HYDRATION, SINGLE_DEPLOYMENT,
      MULTI_DEPLOYMENT]

  Fields:
    deploymentLevel: DeploymentLevel of a blueprint signifies where the
      blueprint will be applied. e.g. [HYDRATION, SINGLE_DEPLOYMENT,
      MULTI_DEPLOYMENT]
    description: The description of the public blueprint.
    displayName: The display name of the public blueprint.
    name: Name of the public blueprint.
    rollbackSupport: Output only. Indicates if the deployment created from
      this blueprint can be rolled back.
    sourceProvider: Source provider is the author of a public blueprint. e.g.
      Google, vendors
  """

    class DeploymentLevelValueValuesEnum(_messages.Enum):
        """DeploymentLevel of a blueprint signifies where the blueprint will be
    applied. e.g. [HYDRATION, SINGLE_DEPLOYMENT, MULTI_DEPLOYMENT]

    Values:
      DEPLOYMENT_LEVEL_UNSPECIFIED: Default unspecified deployment level.
      HYDRATION: Blueprints at HYDRATION level cannot be used to create a
        Deployment (A user cannot manually initate deployment of these
        blueprints on orchestration or workload cluster). These blueprints
        stay in a user's private catalog and are configured and deployed by
        TNA automation.
      SINGLE_DEPLOYMENT: Blueprints at SINGLE_DEPLOYMENT level can be a)
        Modified in private catalog. b) Used to create a deployment on
        orchestration cluster by the user, once approved.
      MULTI_DEPLOYMENT: Blueprints at MULTI_DEPLOYMENT level can be a)
        Modified in private catalog. b) Used to create a deployment on
        orchestration cluster which will create further hydrated deployments.
      WORKLOAD_CLUSTER_DEPLOYMENT: Blueprints at WORKLOAD_CLUSTER_DEPLOYMENT
        level can be a) Modified in private catalog. b) Used to create a
        deployment on workload cluster by the user, once approved.
    """
        DEPLOYMENT_LEVEL_UNSPECIFIED = 0
        HYDRATION = 1
        SINGLE_DEPLOYMENT = 2
        MULTI_DEPLOYMENT = 3
        WORKLOAD_CLUSTER_DEPLOYMENT = 4
    deploymentLevel = _messages.EnumField('DeploymentLevelValueValuesEnum', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    rollbackSupport = _messages.BooleanField(5)
    sourceProvider = _messages.StringField(6)