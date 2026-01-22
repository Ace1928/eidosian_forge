from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def GetActiveRolloutForService(service):
    """Return the latest Rollout for a service.

  This function returns the most recent Rollout that has a status of SUCCESS
  or IN_PROGRESS.

  Args:
    service: The name of the service for which to retrieve the active Rollout.

  Returns:
    The Rollout message corresponding to the active Rollout for the service.
  """
    client = GetClientInstance()
    messages = GetMessagesModule()
    statuses = messages.Rollout.StatusValueValuesEnum
    allowed_statuses = [statuses.SUCCESS, statuses.IN_PROGRESS]
    req = messages.ServicemanagementServicesRolloutsListRequest(serviceName=service)
    result = list(list_pager.YieldFromList(client.services_rollouts, req, predicate=lambda r: r.status in allowed_statuses, limit=1, batch_size_attribute='pageSize', field='rollouts'))
    return result[0] if result else None