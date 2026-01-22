from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDeploymentResourcePoolsQueryDeployedModelsRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsDeploymentResourcePoolsQueryDeployedModelsRequest
  object.

  Fields:
    deploymentResourcePool: Required. The name of the target
      DeploymentResourcePool to query. Format: `projects/{project}/locations/{
      location}/deploymentResourcePools/{deployment_resource_pool}`
    pageSize: The maximum number of DeployedModels to return. The service may
      return fewer than this value.
    pageToken: A page token, received from a previous `QueryDeployedModels`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `QueryDeployedModels` must match the call
      that provided the page token.
  """
    deploymentResourcePool = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)