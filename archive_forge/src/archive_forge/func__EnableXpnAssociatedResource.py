from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def _EnableXpnAssociatedResource(self, host_project, associated_resource, xpn_resource_type):
    """Associate the given resource with the given XPN host project.

    Args:
      host_project: str, ID of the XPN host project
      associated_resource: ID of the resource to associate with host_project
      xpn_resource_type: XpnResourceId.TypeValueValuesEnum, the type of the
         resource
    """
    projects_enable_request = self.messages.ProjectsEnableXpnResourceRequest(xpnResource=self.messages.XpnResourceId(id=associated_resource, type=xpn_resource_type))
    request = self.messages.ComputeProjectsEnableXpnResourceRequest(project=host_project, projectsEnableXpnResourceRequest=projects_enable_request)
    request_tuple = (self.client.projects, 'EnableXpnResource', request)
    msg = 'enable resource [{0}] as an associated resource for project [{1}]'.format(associated_resource, host_project)
    self._MakeRequestSync(request_tuple, msg)