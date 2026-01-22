from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDeploymentResourcePoolsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsDeploymentResourcePoolsCreateRequest
  object.

  Fields:
    googleCloudAiplatformV1CreateDeploymentResourcePoolRequest: A
      GoogleCloudAiplatformV1CreateDeploymentResourcePoolRequest resource to
      be passed as the request body.
    parent: Required. The parent location resource where this
      DeploymentResourcePool will be created. Format:
      `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1CreateDeploymentResourcePoolRequest = _messages.MessageField('GoogleCloudAiplatformV1CreateDeploymentResourcePoolRequest', 1)
    parent = _messages.StringField(2, required=True)