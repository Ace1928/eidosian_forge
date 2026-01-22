from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
class NodeTypesClient(util.VmwareClientBase):
    """cloud vmware node types client."""

    def __init__(self):
        super(NodeTypesClient, self).__init__()
        self.service = self.client.projects_locations_nodeTypes

    def List(self, location_resource):
        request = self.messages.VmwareengineProjectsLocationsNodeTypesListRequest(parent=location_resource.RelativeName())
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='nodeTypes')

    def Get(self, resource):
        request = self.messages.VmwareengineProjectsLocationsNodeTypesGetRequest(name=resource.RelativeName())
        return self.service.Get(request)