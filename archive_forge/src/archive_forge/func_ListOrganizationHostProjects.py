from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def ListOrganizationHostProjects(self, project, organization_id):
    """List the projects in an organization that are enabled as XPN hosts.

    Args:
      project: str, project ID to make the request with.
      organization_id: str, the ID of the organization to list XPN hosts
          for. If None, the organization is inferred from the project.

    Returns:
      Generator for `Project`s corresponding to XPN hosts in the organization.
    """
    request = self.messages.ComputeProjectsListXpnHostsRequest(project=project, projectsListXpnHostsRequest=self.messages.ProjectsListXpnHostsRequest(organization=organization_id))
    return list_pager.YieldFromList(self.client.projects, request, method='ListXpnHosts', batch_size_attribute='maxResults', batch_size=500, field='items')