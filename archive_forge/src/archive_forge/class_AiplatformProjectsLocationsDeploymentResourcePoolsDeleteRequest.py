from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDeploymentResourcePoolsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsDeploymentResourcePoolsDeleteRequest
  object.

  Fields:
    name: Required. The name of the DeploymentResourcePool to delete. Format:
      `projects/{project}/locations/{location}/deploymentResourcePools/{deploy
      ment_resource_pool}`
  """
    name = _messages.StringField(1, required=True)