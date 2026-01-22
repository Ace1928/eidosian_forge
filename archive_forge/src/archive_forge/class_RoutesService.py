from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RoutesService(base_api.BaseApiService):
    """Service class for the routes resource."""
    _NAME = 'routes'

    def __init__(self, client):
        super(ComputeBeta.RoutesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified Route resource.

      Args:
        request: (ComputeRoutesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.routes.delete', ordered_params=['project', 'route'], path_params=['project', 'route'], query_params=['requestId'], relative_path='projects/{project}/global/routes/{route}', request_field='', request_type_name='ComputeRoutesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified Route resource.

      Args:
        request: (ComputeRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Route) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routes.get', ordered_params=['project', 'route'], path_params=['project', 'route'], query_params=[], relative_path='projects/{project}/global/routes/{route}', request_field='', request_type_name='ComputeRoutesGetRequest', response_type_name='Route', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a Route resource in the specified project using the data included in the request.

      Args:
        request: (ComputeRoutesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.routes.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/routes', request_field='route', request_type_name='ComputeRoutesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of Route resources available to the specified project.

      Args:
        request: (ComputeRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RouteList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routes.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/routes', request_field='', request_type_name='ComputeRoutesListRequest', response_type_name='RouteList', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRoutesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.routes.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/routes/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRoutesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)