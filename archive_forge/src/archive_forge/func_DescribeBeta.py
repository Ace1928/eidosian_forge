from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags
def DescribeBeta(self, deployment_resource_pool_ref):
    """Describes a deployment resource pool using v1beta1 API.

    Args:
      deployment_resource_pool_ref: str, Deployment resource pool to describe.

    Returns:
      GoogleCloudAiplatformV1beta1DeploymentResourcePool response message.
    """
    req = self.messages.AiplatformProjectsLocationsDeploymentResourcePoolsGetRequest(name=deployment_resource_pool_ref.RelativeName())
    response = self.client.projects_locations_deploymentResourcePools.Get(req)
    return response