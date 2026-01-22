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
def WaitForDeploymentOperation(client: runapps_v1alpha1_client.RunappsV1alpha1, operation: runapps_v1alpha1_messages.Operation, tracker, tracker_update_func) -> runapps_v1alpha1_messages.Deployment:
    """Waits for an operation to complete.

  Args:
    client: client used to make requests.
    operation: object to wait for.
    tracker: The ProgressTracker that tracks the operation progress.
    tracker_update_func: function to update the tracker on polling.

  Returns:
    the deployment from thex operation.
  """
    return _WaitForOperation(client, operation, client.projects_locations_applications_deployments, tracker, tracker_update_func)