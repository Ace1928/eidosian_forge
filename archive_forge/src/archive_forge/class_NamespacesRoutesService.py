from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class NamespacesRoutesService(base_api.BaseApiService):
    """Service class for the namespaces_routes resource."""
    _NAME = 'namespaces_routes'

    def __init__(self, client):
        super(RunV1.NamespacesRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get information about a route.

      Args:
        request: (RunNamespacesRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Route) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/serving.knative.dev/v1/namespaces/{namespacesId}/routes/{routesId}', http_method='GET', method_id='run.namespaces.routes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/serving.knative.dev/v1/{+name}', request_field='', request_type_name='RunNamespacesRoutesGetRequest', response_type_name='Route', supports_download=False)

    def List(self, request, global_params=None):
        """List routes.

      Args:
        request: (RunNamespacesRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/serving.knative.dev/v1/namespaces/{namespacesId}/routes', http_method='GET', method_id='run.namespaces.routes.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/serving.knative.dev/v1/{+parent}/routes', request_field='', request_type_name='RunNamespacesRoutesListRequest', response_type_name='ListRoutesResponse', supports_download=False)