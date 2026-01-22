from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def ListPeeredDnsDomains(project_number, service, network):
    """Make API call to list the peered DNS domains for a specific connection.

  Args:
    project_number: The number of the project which is peered with the service.
    service: The name of the service to list the peered DNS domains for.
    network: The network in the consumer project peered with the service.

  Raises:
    exceptions.ListPeeredDnsDomainsPermissionDeniedException: when the delete
    peered DNS domain API fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the peering
    service.

  Returns:
    The list of peered DNS domains.
  """
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    request = messages.ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsListRequest(parent='services/%s/projects/%s/global/networks/%s' % (service, project_number, network))
    try:
        return client.services_projects_global_networks_peeredDnsDomains.List(request).peeredDnsDomains
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.ListPeeredDnsDomainsPermissionDeniedException)