from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from typing import List, Optional
from apitools.base.py import encoding as apitools_encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import retry
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetDeployment(client: runapps_v1alpha1_client.RunappsV1alpha1, deployment_name: str) -> Optional[runapps_v1alpha1_messages.Deployment]:
    """Calls GetDeployment API of Runapps.

  Args:
    client: the api client to use.
    deployment_name: the canonical name of the deployment.  For example:
      projects/<project>/locations/<location>/applications/<app>/deployment/<id>

  Returns:
    the Deployment object.  None is returned if the deployment cannot be found.
  """
    try:
        return client.projects_locations_applications_deployments.Get(client.MESSAGES_MODULE.RunappsProjectsLocationsApplicationsDeploymentsGetRequest(name=deployment_name))
    except apitools_exceptions.HttpNotFoundError:
        return None