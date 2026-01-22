from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.eventflow.v1beta2 import eventflow_v1beta2_messages as messages
class ProjectsFlowsService(base_api.BaseApiService):
    """Service class for the projects_flows resource."""
    _NAME = 'projects_flows'

    def __init__(self, client):
        super(EventflowV1beta2.ProjectsFlowsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a flow, and returns the new Flow.

      Args:
        request: (EventflowProjectsFlowsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Flow) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta2/projects/{projectsId}/flows', http_method='POST', method_id='eventflow.projects.flows.create', ordered_params=['namespace'], path_params=['namespace'], query_params=[], relative_path='v1beta2/projects/{+namespace}/flows', request_field='flow', request_type_name='EventflowProjectsFlowsCreateRequest', response_type_name='Flow', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a flow. Returns NOT_FOUND if the item does not exist.

      Args:
        request: (EventflowProjectsFlowsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta2/projects/{projectsId}/flows/{flowsId}', http_method='DELETE', method_id='eventflow.projects.flows.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta2/{+name}', request_field='', request_type_name='EventflowProjectsFlowsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a flow. Returns NOT_FOUND if the flow does not exist.

      Args:
        request: (EventflowProjectsFlowsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Flow) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta2/projects/{projectsId}/flows/{flowsId}', http_method='GET', method_id='eventflow.projects.flows.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta2/{+name}', request_field='', request_type_name='EventflowProjectsFlowsGetRequest', response_type_name='Flow', supports_download=False)

    def List(self, request, global_params=None):
        """Lists flows.

      Args:
        request: (EventflowProjectsFlowsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFlowsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta2/projects/{projectsId}/flows', http_method='GET', method_id='eventflow.projects.flows.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta2/{+parent}/flows', request_field='', request_type_name='EventflowProjectsFlowsListRequest', response_type_name='ListFlowsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a flow, returning the updated flow. Empty fields (proto3 default values) mean don't change those fields. The call returns INVALID_ARGUMENT status if the spec.name, spec.namespace, or spec.trigger.event_type is change. trigger.event_type is changed.

      Args:
        request: (EventflowProjectsFlowsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Flow) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta2/projects/{projectsId}/flows/{flowsId}', http_method='PUT', method_id='eventflow.projects.flows.update', ordered_params=['namespace', 'name'], path_params=['name', 'namespace'], query_params=[], relative_path='v1beta2/projects/{+namespace}/flows/{+name}', request_field='flow', request_type_name='EventflowProjectsFlowsUpdateRequest', response_type_name='Flow', supports_download=False)