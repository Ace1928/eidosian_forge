import collections
import copy
import enum
import sys
from typing import List
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import http_retry
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
def BatchGetService(parent, services):
    """Make API call to get service state for multiple services .

  Args:
    parent: Parent resource to get service state for. format-"projects/100",
      "folders/101" or "organizations/102".
    services: Services. Current supported value:(format:
      "{resource}/{resource_Id}/services/{service}").

  Raises:
    exceptions.BatchGetServicePermissionDeniedException: when getting batch
      service state for services in the resource.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    Service state of the given resource.
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    request = messages.ServiceusageServicesBatchGetRequest(parent=parent, services=services, view=messages.ServiceusageServicesBatchGetRequest.ViewValueValuesEnum.SERVICE_STATE_VIEW_FULL)
    try:
        return client.services.BatchGet(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.BatchGetServicePermissionDeniedException)