from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsTlsInspectionPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_tlsInspectionPolicies resource."""
    _NAME = 'projects_locations_tlsInspectionPolicies'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsTlsInspectionPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new TlsInspectionPolicy in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsTlsInspectionPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tlsInspectionPolicies', http_method='POST', method_id='networksecurity.projects.locations.tlsInspectionPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['tlsInspectionPolicyId'], relative_path='v1/{+parent}/tlsInspectionPolicies', request_field='tlsInspectionPolicy', request_type_name='NetworksecurityProjectsLocationsTlsInspectionPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single TlsInspectionPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsTlsInspectionPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tlsInspectionPolicies/{tlsInspectionPoliciesId}', http_method='DELETE', method_id='networksecurity.projects.locations.tlsInspectionPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsTlsInspectionPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single TlsInspectionPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsTlsInspectionPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TlsInspectionPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tlsInspectionPolicies/{tlsInspectionPoliciesId}', http_method='GET', method_id='networksecurity.projects.locations.tlsInspectionPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsTlsInspectionPoliciesGetRequest', response_type_name='TlsInspectionPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TlsInspectionPolicies in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsTlsInspectionPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTlsInspectionPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tlsInspectionPolicies', http_method='GET', method_id='networksecurity.projects.locations.tlsInspectionPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/tlsInspectionPolicies', request_field='', request_type_name='NetworksecurityProjectsLocationsTlsInspectionPoliciesListRequest', response_type_name='ListTlsInspectionPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single TlsInspectionPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsTlsInspectionPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tlsInspectionPolicies/{tlsInspectionPoliciesId}', http_method='PATCH', method_id='networksecurity.projects.locations.tlsInspectionPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='tlsInspectionPolicy', request_type_name='NetworksecurityProjectsLocationsTlsInspectionPoliciesPatchRequest', response_type_name='Operation', supports_download=False)