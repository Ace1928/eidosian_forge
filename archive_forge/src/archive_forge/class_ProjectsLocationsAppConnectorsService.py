from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class ProjectsLocationsAppConnectorsService(base_api.BaseApiService):
    """Service class for the projects_locations_appConnectors resource."""
    _NAME = 'projects_locations_appConnectors'

    def __init__(self, client):
        super(BeyondcorpV1alpha.ProjectsLocationsAppConnectorsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AppConnector in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors', http_method='POST', method_id='beyondcorp.projects.locations.appConnectors.create', ordered_params=['parent'], path_params=['parent'], query_params=['appConnectorId', 'requestId', 'validateOnly'], relative_path='v1alpha/{+parent}/appConnectors', request_field='googleCloudBeyondcorpAppconnectorsV1alphaAppConnector', request_type_name='BeyondcorpProjectsLocationsAppConnectorsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single AppConnector.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}', http_method='DELETE', method_id='beyondcorp.projects.locations.appConnectors.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectorsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single AppConnector.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnector) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}', http_method='GET', method_id='beyondcorp.projects.locations.appConnectors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectorsGetRequest', response_type_name='GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnector', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}:getIamPolicy', http_method='GET', method_id='beyondcorp.projects.locations.appConnectors.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectorsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists AppConnectors in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpAppconnectorsV1alphaListAppConnectorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors', http_method='GET', method_id='beyondcorp.projects.locations.appConnectors.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/appConnectors', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectorsListRequest', response_type_name='GoogleCloudBeyondcorpAppconnectorsV1alphaListAppConnectorsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single AppConnector.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}', http_method='PATCH', method_id='beyondcorp.projects.locations.appConnectors.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='googleCloudBeyondcorpAppconnectorsV1alphaAppConnector', request_type_name='BeyondcorpProjectsLocationsAppConnectorsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ReportStatus(self, request, global_params=None):
        """Report status for a given connector.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsReportStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ReportStatus')
        return self._RunMethod(config, request, global_params=global_params)
    ReportStatus.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}:reportStatus', http_method='POST', method_id='beyondcorp.projects.locations.appConnectors.reportStatus', ordered_params=['appConnector'], path_params=['appConnector'], query_params=[], relative_path='v1alpha/{+appConnector}:reportStatus', request_field='googleCloudBeyondcorpAppconnectorsV1alphaReportStatusRequest', request_type_name='BeyondcorpProjectsLocationsAppConnectorsReportStatusRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ResolveInstanceConfig(self, request, global_params=None):
        """Gets instance configuration for a given AppConnector. An internal method called by a AppConnector to get its container config.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsResolveInstanceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpAppconnectorsV1alphaResolveInstanceConfigResponse) The response message.
      """
        config = self.GetMethodConfig('ResolveInstanceConfig')
        return self._RunMethod(config, request, global_params=global_params)
    ResolveInstanceConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}:resolveInstanceConfig', http_method='GET', method_id='beyondcorp.projects.locations.appConnectors.resolveInstanceConfig', ordered_params=['appConnector'], path_params=['appConnector'], query_params=[], relative_path='v1alpha/{+appConnector}:resolveInstanceConfig', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectorsResolveInstanceConfigRequest', response_type_name='GoogleCloudBeyondcorpAppconnectorsV1alphaResolveInstanceConfigResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}:setIamPolicy', http_method='POST', method_id='beyondcorp.projects.locations.appConnectors.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='BeyondcorpProjectsLocationsAppConnectorsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectorsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnectors/{appConnectorsId}:testIamPermissions', http_method='POST', method_id='beyondcorp.projects.locations.appConnectors.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='BeyondcorpProjectsLocationsAppConnectorsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)