from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1alpha1 import containeranalysis_v1alpha1_messages as messages
class ProjectsOperationsService(base_api.BaseApiService):
    """Service class for the projects_operations resource."""
    _NAME = 'projects_operations'

    def __init__(self, client):
        super(ContaineranalysisV1alpha1.ProjectsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `Operation`.

      Args:
        request: (ContaineranalysisProjectsOperationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/operations', http_method='POST', method_id='containeranalysis.projects.operations.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/operations', request_field='createOperationRequest', request_type_name='ContaineranalysisProjectsOperationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing operation returns an error if operation does not exist. The only valid operations are to update mark the done bit change the result.

      Args:
        request: (ContaineranalysisProjectsOperationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/operations/{operationsId}', http_method='PATCH', method_id='containeranalysis.projects.operations.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='updateOperationRequest', request_type_name='ContaineranalysisProjectsOperationsPatchRequest', response_type_name='Operation', supports_download=False)