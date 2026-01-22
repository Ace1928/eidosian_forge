from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import json
from typing import List, MutableSequence, Optional
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run.integrations import api_utils
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import integration_list_printer
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import stages
from googlecloudsdk.command_lib.run.integrations import typekits_util
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
import six
def _GetStatusFromLatestDeployment(self, deployment: str, deployment_cache: {str, runapps_v1alpha1_messages.Deployment}) -> runapps_v1alpha1_messages.DeploymentStatus.StateValueValuesEnum:
    """Get status from latest deployment.

    Args:
      deployment: the name of the latest deployment
      deployment_cache: a map of cached deployments

    Returns:
      status of the latest deployment.
    """
    status = runapps_v1alpha1_messages.DeploymentStatus.StateValueValuesEnum.STATE_UNSPECIFIED
    if not deployment:
        return status
    if deployment_cache.get(deployment):
        status = deployment_cache[deployment].status.state
    else:
        dep = api_utils.GetDeployment(self._client, deployment)
        if dep:
            status = dep.status.state
            deployment_cache[deployment] = dep
    return status