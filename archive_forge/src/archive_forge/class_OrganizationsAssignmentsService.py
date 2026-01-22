from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha1 import osconfig_v1alpha1_messages as messages
class OrganizationsAssignmentsService(base_api.BaseApiService):
    """Service class for the organizations_assignments resource."""
    _NAME = 'organizations_assignments'

    def __init__(self, client):
        super(OsconfigV1alpha1.OrganizationsAssignmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an OS Config Assignment.

      Args:
        request: (OsconfigOrganizationsAssignmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Assignment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/assignments', http_method='POST', method_id='osconfig.organizations.assignments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/assignments', request_field='assignment', request_type_name='OsconfigOrganizationsAssignmentsCreateRequest', response_type_name='Assignment', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an OS Config Assignment.

      Args:
        request: (OsconfigOrganizationsAssignmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/assignments/{assignmentsId}', http_method='DELETE', method_id='osconfig.organizations.assignments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigOrganizationsAssignmentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get an OS Config Assignment.

      Args:
        request: (OsconfigOrganizationsAssignmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Assignment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/assignments/{assignmentsId}', http_method='GET', method_id='osconfig.organizations.assignments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='OsconfigOrganizationsAssignmentsGetRequest', response_type_name='Assignment', supports_download=False)

    def List(self, request, global_params=None):
        """Get a page of OS Config Assignments.

      Args:
        request: (OsconfigOrganizationsAssignmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAssignmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/assignments', http_method='GET', method_id='osconfig.organizations.assignments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/assignments', request_field='', request_type_name='OsconfigOrganizationsAssignmentsListRequest', response_type_name='ListAssignmentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an OS Config Assignment.

      Args:
        request: (OsconfigOrganizationsAssignmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Assignment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/assignments/{assignmentsId}', http_method='PATCH', method_id='osconfig.organizations.assignments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='assignment', request_type_name='OsconfigOrganizationsAssignmentsPatchRequest', response_type_name='Assignment', supports_download=False)