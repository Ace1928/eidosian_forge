from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class NamespacesExecutionsService(base_api.BaseApiService):
    """Service class for the namespaces_executions resource."""
    _NAME = 'namespaces_executions'

    def __init__(self, client):
        super(RunV1.NamespacesExecutionsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancel an execution.

      Args:
        request: (RunNamespacesExecutionsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Execution) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/run.googleapis.com/v1/namespaces/{namespacesId}/executions/{executionsId}:cancel', http_method='POST', method_id='run.namespaces.executions.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/run.googleapis.com/v1/{+name}:cancel', request_field='cancelExecutionRequest', request_type_name='RunNamespacesExecutionsCancelRequest', response_type_name='Execution', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an execution.

      Args:
        request: (RunNamespacesExecutionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Status) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/run.googleapis.com/v1/namespaces/{namespacesId}/executions/{executionsId}', http_method='DELETE', method_id='run.namespaces.executions.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'kind', 'propagationPolicy'], relative_path='apis/run.googleapis.com/v1/{+name}', request_field='', request_type_name='RunNamespacesExecutionsDeleteRequest', response_type_name='Status', supports_download=False)

    def Get(self, request, global_params=None):
        """Get information about an execution.

      Args:
        request: (RunNamespacesExecutionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Execution) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/run.googleapis.com/v1/namespaces/{namespacesId}/executions/{executionsId}', http_method='GET', method_id='run.namespaces.executions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/run.googleapis.com/v1/{+name}', request_field='', request_type_name='RunNamespacesExecutionsGetRequest', response_type_name='Execution', supports_download=False)

    def List(self, request, global_params=None):
        """List executions.

      Args:
        request: (RunNamespacesExecutionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExecutionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/run.googleapis.com/v1/namespaces/{namespacesId}/executions', http_method='GET', method_id='run.namespaces.executions.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/run.googleapis.com/v1/{+parent}/executions', request_field='', request_type_name='RunNamespacesExecutionsListRequest', response_type_name='ListExecutionsResponse', supports_download=False)