from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDeploymentResourcePoolsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsDeploymentResourcePoolsGetRequest object.

  Fields:
    name: Required. The name of the DeploymentResourcePool to retrieve.
      Format: `projects/{project}/locations/{location}/deploymentResourcePools
      /{deployment_resource_pool}`
  """
    name = _messages.StringField(1, required=True)