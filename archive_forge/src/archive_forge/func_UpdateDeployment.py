from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
def UpdateDeployment(deployment, deployment_full_name):
    """Calls into the UpdateDeployment API.

  Args:
    deployment: a messages.Deployment resource (containing properties like the
      blueprint).
    deployment_full_name: the fully qualified name of the deployment.

  Returns:
    A messages.OperationMetadata representing a long-running operation.
  """
    client = GetClientInstance()
    messages = client.MESSAGES_MODULE
    return client.projects_locations_deployments.Patch(messages.ConfigProjectsLocationsDeploymentsPatchRequest(deployment=deployment, name=deployment_full_name, updateMask=None))