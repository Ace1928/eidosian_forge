from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha1 import osconfig_v1alpha1_messages as messages
class ProjectsAssignmentsService(base_api.BaseApiService):
    """Service class for the projects_assignments resource."""
    _NAME = 'projects_assignments'

    def __init__(self, client):
        super(OsconfigV1alpha1.ProjectsAssignmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an OS Config Assignment.

      Args:
        request: (OsconfigProjectsAssignmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Assignment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/assignments', http_method='POST', method_id='osconfig.projects.assignments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/assignments', request_field='assignment', request_type_name='OsconfigProjectsAssignmentsCreateRequest', response_type_name='Assignment', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an OS Config Assignment.

      Args:
        request: (OsconfigProjectsAssignmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/assignments/{assignmentsId}', http_method='DELETE', method_id='osconfig.projects.assignments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigProjectsAssignmentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get an OS Config Assignment.

      Args:
        request: (OsconfigProjectsAssignmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Assignment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/assignments/{assignmentsId}', http_method='GET', method_id='osconfig.projects.assignments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigProjectsAssignmentsGetRequest', response_type_name='Assignment', supports_download=False)

    def List(self, request, global_params=None):
        """Get a page of OS Config Assignments.

      Args:
        request: (OsconfigProjectsAssignmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAssignmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/assignments', http_method='GET', method_id='osconfig.projects.assignments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/assignments', request_field='', request_type_name='OsconfigProjectsAssignmentsListRequest', response_type_name='ListAssignmentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an OS Config Assignment.

      Args:
        request: (OsconfigProjectsAssignmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Assignment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/assignments/{assignmentsId}', http_method='PATCH', method_id='osconfig.projects.assignments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='assignment', request_type_name='OsconfigProjectsAssignmentsPatchRequest', response_type_name='Assignment', supports_download=False)