from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class TargetSslProxiesService(base_api.BaseApiService):
    """Service class for the targetSslProxies resource."""
    _NAME = 'targetSslProxies'

    def __init__(self, client):
        super(ComputeBeta.TargetSslProxiesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetSslProxy resource.

      Args:
        request: (ComputeTargetSslProxiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.targetSslProxies.delete', ordered_params=['project', 'targetSslProxy'], path_params=['project', 'targetSslProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetSslProxies/{targetSslProxy}', request_field='', request_type_name='ComputeTargetSslProxiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetSslProxy resource.

      Args:
        request: (ComputeTargetSslProxiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetSslProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetSslProxies.get', ordered_params=['project', 'targetSslProxy'], path_params=['project', 'targetSslProxy'], query_params=[], relative_path='projects/{project}/global/targetSslProxies/{targetSslProxy}', request_field='', request_type_name='ComputeTargetSslProxiesGetRequest', response_type_name='TargetSslProxy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetSslProxy resource in the specified project using the data included in the request.

      Args:
        request: (ComputeTargetSslProxiesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetSslProxies.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/targetSslProxies', request_field='targetSslProxy', request_type_name='ComputeTargetSslProxiesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of TargetSslProxy resources available to the specified project.

      Args:
        request: (ComputeTargetSslProxiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetSslProxyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetSslProxies.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/targetSslProxies', request_field='', request_type_name='ComputeTargetSslProxiesListRequest', response_type_name='TargetSslProxyList', supports_download=False)

    def SetBackendService(self, request, global_params=None):
        """Changes the BackendService for TargetSslProxy.

      Args:
        request: (ComputeTargetSslProxiesSetBackendServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetBackendService')
        return self._RunMethod(config, request, global_params=global_params)
    SetBackendService.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetSslProxies.setBackendService', ordered_params=['project', 'targetSslProxy'], path_params=['project', 'targetSslProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetSslProxies/{targetSslProxy}/setBackendService', request_field='targetSslProxiesSetBackendServiceRequest', request_type_name='ComputeTargetSslProxiesSetBackendServiceRequest', response_type_name='Operation', supports_download=False)

    def SetCertificateMap(self, request, global_params=None):
        """Changes the Certificate Map for TargetSslProxy.

      Args:
        request: (ComputeTargetSslProxiesSetCertificateMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetCertificateMap')
        return self._RunMethod(config, request, global_params=global_params)
    SetCertificateMap.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetSslProxies.setCertificateMap', ordered_params=['project', 'targetSslProxy'], path_params=['project', 'targetSslProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetSslProxies/{targetSslProxy}/setCertificateMap', request_field='targetSslProxiesSetCertificateMapRequest', request_type_name='ComputeTargetSslProxiesSetCertificateMapRequest', response_type_name='Operation', supports_download=False)

    def SetProxyHeader(self, request, global_params=None):
        """Changes the ProxyHeaderType for TargetSslProxy.

      Args:
        request: (ComputeTargetSslProxiesSetProxyHeaderRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetProxyHeader')
        return self._RunMethod(config, request, global_params=global_params)
    SetProxyHeader.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetSslProxies.setProxyHeader', ordered_params=['project', 'targetSslProxy'], path_params=['project', 'targetSslProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetSslProxies/{targetSslProxy}/setProxyHeader', request_field='targetSslProxiesSetProxyHeaderRequest', request_type_name='ComputeTargetSslProxiesSetProxyHeaderRequest', response_type_name='Operation', supports_download=False)

    def SetSslCertificates(self, request, global_params=None):
        """Changes SslCertificates for TargetSslProxy.

      Args:
        request: (ComputeTargetSslProxiesSetSslCertificatesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSslCertificates')
        return self._RunMethod(config, request, global_params=global_params)
    SetSslCertificates.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetSslProxies.setSslCertificates', ordered_params=['project', 'targetSslProxy'], path_params=['project', 'targetSslProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetSslProxies/{targetSslProxy}/setSslCertificates', request_field='targetSslProxiesSetSslCertificatesRequest', request_type_name='ComputeTargetSslProxiesSetSslCertificatesRequest', response_type_name='Operation', supports_download=False)

    def SetSslPolicy(self, request, global_params=None):
        """Sets the SSL policy for TargetSslProxy. The SSL policy specifies the server-side support for SSL features. This affects connections between clients and the load balancer. They do not affect the connection between the load balancer and the backends.

      Args:
        request: (ComputeTargetSslProxiesSetSslPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSslPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetSslPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetSslProxies.setSslPolicy', ordered_params=['project', 'targetSslProxy'], path_params=['project', 'targetSslProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetSslProxies/{targetSslProxy}/setSslPolicy', request_field='sslPolicyReference', request_type_name='ComputeTargetSslProxiesSetSslPolicyRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeTargetSslProxiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetSslProxies.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/targetSslProxies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeTargetSslProxiesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)