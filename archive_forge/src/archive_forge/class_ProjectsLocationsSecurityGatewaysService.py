from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class ProjectsLocationsSecurityGatewaysService(base_api.BaseApiService):
    """Service class for the projects_locations_securityGateways resource."""
    _NAME = 'projects_locations_securityGateways'

    def __init__(self, client):
        super(BeyondcorpV1alpha.ProjectsLocationsSecurityGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new SecurityGateway in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsSecurityGatewaysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/securityGateways', http_method='POST', method_id='beyondcorp.projects.locations.securityGateways.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'securityGatewayId'], relative_path='v1alpha/{+parent}/securityGateways', request_field='googleCloudBeyondcorpSecuritygatewaysV1alphaSecurityGateway', request_type_name='BeyondcorpProjectsLocationsSecurityGatewaysCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single SecurityGateway.

      Args:
        request: (BeyondcorpProjectsLocationsSecurityGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/securityGateways/{securityGatewaysId}', http_method='DELETE', method_id='beyondcorp.projects.locations.securityGateways.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsSecurityGatewaysDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single SecurityGateway.

      Args:
        request: (BeyondcorpProjectsLocationsSecurityGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSecuritygatewaysV1alphaSecurityGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/securityGateways/{securityGatewaysId}', http_method='GET', method_id='beyondcorp.projects.locations.securityGateways.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsSecurityGatewaysGetRequest', response_type_name='GoogleCloudBeyondcorpSecuritygatewaysV1alphaSecurityGateway', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SecurityGateways in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsSecurityGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSecuritygatewaysV1alphaListSecurityGatewaysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/securityGateways', http_method='GET', method_id='beyondcorp.projects.locations.securityGateways.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/securityGateways', request_field='', request_type_name='BeyondcorpProjectsLocationsSecurityGatewaysListRequest', response_type_name='GoogleCloudBeyondcorpSecuritygatewaysV1alphaListSecurityGatewaysResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single SecurityGateway.

      Args:
        request: (BeyondcorpProjectsLocationsSecurityGatewaysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/securityGateways/{securityGatewaysId}', http_method='PATCH', method_id='beyondcorp.projects.locations.securityGateways.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='googleCloudBeyondcorpSecuritygatewaysV1alphaSecurityGateway', request_type_name='BeyondcorpProjectsLocationsSecurityGatewaysPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)