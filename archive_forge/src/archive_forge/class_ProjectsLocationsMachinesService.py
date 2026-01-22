from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgecontainer.v1beta import edgecontainer_v1beta_messages as messages
class ProjectsLocationsMachinesService(base_api.BaseApiService):
    """Service class for the projects_locations_machines resource."""
    _NAME = 'projects_locations_machines'

    def __init__(self, client):
        super(EdgecontainerV1beta.ProjectsLocationsMachinesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single Machine.

      Args:
        request: (EdgecontainerProjectsLocationsMachinesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Machine) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/machines/{machinesId}', http_method='GET', method_id='edgecontainer.projects.locations.machines.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='EdgecontainerProjectsLocationsMachinesGetRequest', response_type_name='Machine', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Machines in a given project and location.

      Args:
        request: (EdgecontainerProjectsLocationsMachinesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMachinesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/machines', http_method='GET', method_id='edgecontainer.projects.locations.machines.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/machines', request_field='', request_type_name='EdgecontainerProjectsLocationsMachinesListRequest', response_type_name='ListMachinesResponse', supports_download=False)