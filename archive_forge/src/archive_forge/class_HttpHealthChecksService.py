from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class HttpHealthChecksService(base_api.BaseApiService):
    """Service class for the httpHealthChecks resource."""
    _NAME = 'httpHealthChecks'

    def __init__(self, client):
        super(ComputeBeta.HttpHealthChecksService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified HttpHealthCheck resource.

      Args:
        request: (ComputeHttpHealthChecksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.httpHealthChecks.delete', ordered_params=['project', 'httpHealthCheck'], path_params=['httpHealthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/httpHealthChecks/{httpHealthCheck}', request_field='', request_type_name='ComputeHttpHealthChecksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified HttpHealthCheck resource.

      Args:
        request: (ComputeHttpHealthChecksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpHealthCheck) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.httpHealthChecks.get', ordered_params=['project', 'httpHealthCheck'], path_params=['httpHealthCheck', 'project'], query_params=[], relative_path='projects/{project}/global/httpHealthChecks/{httpHealthCheck}', request_field='', request_type_name='ComputeHttpHealthChecksGetRequest', response_type_name='HttpHealthCheck', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a HttpHealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeHttpHealthChecksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.httpHealthChecks.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/httpHealthChecks', request_field='httpHealthCheck', request_type_name='ComputeHttpHealthChecksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of HttpHealthCheck resources available to the specified project.

      Args:
        request: (ComputeHttpHealthChecksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpHealthCheckList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.httpHealthChecks.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/httpHealthChecks', request_field='', request_type_name='ComputeHttpHealthChecksListRequest', response_type_name='HttpHealthCheckList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a HttpHealthCheck resource in the specified project using the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeHttpHealthChecksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.httpHealthChecks.patch', ordered_params=['project', 'httpHealthCheck'], path_params=['httpHealthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/httpHealthChecks/{httpHealthCheck}', request_field='httpHealthCheckResource', request_type_name='ComputeHttpHealthChecksPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeHttpHealthChecksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.httpHealthChecks.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/httpHealthChecks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeHttpHealthChecksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a HttpHealthCheck resource in the specified project using the data included in the request.

      Args:
        request: (ComputeHttpHealthChecksUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.httpHealthChecks.update', ordered_params=['project', 'httpHealthCheck'], path_params=['httpHealthCheck', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/httpHealthChecks/{httpHealthCheck}', request_field='httpHealthCheckResource', request_type_name='ComputeHttpHealthChecksUpdateRequest', response_type_name='Operation', supports_download=False)