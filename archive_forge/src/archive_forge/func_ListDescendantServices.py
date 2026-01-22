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
def ListDescendantServices(resource: str, service_group: str, page_size: int=50):
    """Make API call to list descendant services of a specific service group.

  Args:
    resource: The target resource in the format:
      '{resource_type}/{resource_name}'.
    service_group: Service group, for example,
      'services/compute.googleapis.com/groups/dependencies'.
    page_size: The page size to list. The default page_size is 50.

  Raises:
    exceptions.ListDescendantServicesPermissionDeniedException: when listing
      descendant services fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    Descendant services in the given service group.
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    request = messages.ServiceusageServicesGroupsDescendantServicesListRequest(parent='{}/{}'.format(resource, service_group))
    try:
        return list_pager.YieldFromList(_Lister(client.services_groups_descendantServices), request, batch_size_attribute='pageSize', batch_size=page_size, field='services')
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.ListDescendantServicesPermissionDeniedException)