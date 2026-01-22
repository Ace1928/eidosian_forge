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
def ListAncestorGroups(resource: str, service: str, page_size=50):
    """Make API call to list ancestor groups that depend on the service.

  Args:
    resource: The target resource.format : '{resource_type}/{resource_name}'.
    service: The identifier of the service to get ancestor groups of, for
      example, 'services/compute.googleapis.com'.
    page_size: The page size to list.The default page_size is 50.

  Raises:
    exceptions.ListAncestorGroupsPermissionDeniedException: when listing
      ancestor group fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    Ancestor groups that depend on the service.
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    request = messages.ServiceusageServicesAncestorGroupsListRequest(name=f'{resource}/{service}')
    try:
        return list_pager.YieldFromList(_Lister(client.services_ancestorGroups), request, batch_size_attribute='pageSize', batch_size=page_size, field='groups')
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.ListAncestorGroupsPermissionDeniedException)