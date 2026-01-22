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
def ListGroupMembersV2Alpha(resource: str, service_group: str, page_size: int=50, limit: int=sys.maxsize):
    """Make API call to list group members of a specific service group.

  Args:
    resource: The target resource.
    service_group: Service group which owns a collection of group members, for
      example, 'services/compute.googleapis.com/groups/dependencies'.
    page_size: The page size to list. The default page_size is 50.
    limit: The max number of services to display.

  Raises:
    exceptions.ListGroupMembersPermissionDeniedException: when listing
      group members fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    Group members in the given service group.
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    request = messages.ServiceusageServicesGroupsMembersListRequest(parent=resource + '/' + service_group)
    try:
        return list_pager.YieldFromList(_Lister(client.services_groups_members), request, limit=limit, batch_size_attribute='pageSize', batch_size=page_size, field='memberStates')
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.ListGroupMembersPermissionDeniedException)