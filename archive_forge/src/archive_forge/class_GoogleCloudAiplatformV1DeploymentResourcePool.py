from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeploymentResourcePool(_messages.Message):
    """A description of resources that can be shared by multiple
  DeployedModels, whose underlying specification consists of a
  DedicatedResources.

  Fields:
    createTime: Output only. Timestamp when this DeploymentResourcePool was
      created.
    dedicatedResources: Required. The underlying DedicatedResources that the
      DeploymentResourcePool uses.
    name: Immutable. The resource name of the DeploymentResourcePool. Format:
      `projects/{project}/locations/{location}/deploymentResourcePools/{deploy
      ment_resource_pool}`
  """
    createTime = _messages.StringField(1)
    dedicatedResources = _messages.MessageField('GoogleCloudAiplatformV1DedicatedResources', 2)
    name = _messages.StringField(3)