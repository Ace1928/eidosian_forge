from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionBackendServicesService(base_api.BaseApiService):
    """Service class for the regionBackendServices resource."""
    _NAME = 'regionBackendServices'

    def __init__(self, client):
        super(ComputeBeta.RegionBackendServicesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified regional BackendService resource.

      Args:
        request: (ComputeRegionBackendServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionBackendServices.delete', ordered_params=['project', 'region', 'backendService'], path_params=['backendService', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/backendServices/{backendService}', request_field='', request_type_name='ComputeRegionBackendServicesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified regional BackendService resource.

      Args:
        request: (ComputeRegionBackendServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendService) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionBackendServices.get', ordered_params=['project', 'region', 'backendService'], path_params=['backendService', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/backendServices/{backendService}', request_field='', request_type_name='ComputeRegionBackendServicesGetRequest', response_type_name='BackendService', supports_download=False)

    def GetHealth(self, request, global_params=None):
        """Gets the most recent health check results for this regional BackendService.

      Args:
        request: (ComputeRegionBackendServicesGetHealthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendServiceGroupHealth) The response message.
      """
        config = self.GetMethodConfig('GetHealth')
        return self._RunMethod(config, request, global_params=global_params)
    GetHealth.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionBackendServices.getHealth', ordered_params=['project', 'region', 'backendService'], path_params=['backendService', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/backendServices/{backendService}/getHealth', request_field='resourceGroupReference', request_type_name='ComputeRegionBackendServicesGetHealthRequest', response_type_name='BackendServiceGroupHealth', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeRegionBackendServicesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionBackendServices.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/backendServices/{resource}/getIamPolicy', request_field='', request_type_name='ComputeRegionBackendServicesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a regional BackendService resource in the specified project using the data included in the request. For more information, see Backend services overview.

      Args:
        request: (ComputeRegionBackendServicesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionBackendServices.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/backendServices', request_field='backendService', request_type_name='ComputeRegionBackendServicesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of regional BackendService resources available to the specified project in the given region.

      Args:
        request: (ComputeRegionBackendServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendServiceList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionBackendServices.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/backendServices', request_field='', request_type_name='ComputeRegionBackendServicesListRequest', response_type_name='BackendServiceList', supports_download=False)

    def ListUsable(self, request, global_params=None):
        """Retrieves an aggregated list of all usable backend services in the specified project in the given region.

      Args:
        request: (ComputeRegionBackendServicesListUsableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendServiceListUsable) The response message.
      """
        config = self.GetMethodConfig('ListUsable')
        return self._RunMethod(config, request, global_params=global_params)
    ListUsable.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionBackendServices.listUsable', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/backendServices/listUsable', request_field='', request_type_name='ComputeRegionBackendServicesListUsableRequest', response_type_name='BackendServiceListUsable', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified regional BackendService resource with the data included in the request. For more information, see Understanding backend services This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeRegionBackendServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionBackendServices.patch', ordered_params=['project', 'region', 'backendService'], path_params=['backendService', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/backendServices/{backendService}', request_field='backendServiceResource', request_type_name='ComputeRegionBackendServicesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeRegionBackendServicesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionBackendServices.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/backendServices/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeRegionBackendServicesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetSecurityPolicy(self, request, global_params=None):
        """Sets the Google Cloud Armor security policy for the specified backend service. For more information, see Google Cloud Armor Overview.

      Args:
        request: (ComputeRegionBackendServicesSetSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSecurityPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetSecurityPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionBackendServices.setSecurityPolicy', ordered_params=['project', 'region', 'backendService'], path_params=['backendService', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/backendServices/{backendService}/setSecurityPolicy', request_field='securityPolicyReference', request_type_name='ComputeRegionBackendServicesSetSecurityPolicyRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionBackendServicesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionBackendServices.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/backendServices/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionBackendServicesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified regional BackendService resource with the data included in the request. For more information, see Backend services overview .

      Args:
        request: (ComputeRegionBackendServicesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.regionBackendServices.update', ordered_params=['project', 'region', 'backendService'], path_params=['backendService', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/backendServices/{backendService}', request_field='backendServiceResource', request_type_name='ComputeRegionBackendServicesUpdateRequest', response_type_name='Operation', supports_download=False)