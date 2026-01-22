from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class GlobalOperationsService(base_api.BaseApiService):
    """Service class for the globalOperations resource."""
    _NAME = 'globalOperations'

    def __init__(self, client):
        super(ComputeBeta.GlobalOperationsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of all operations. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeGlobalOperationsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OperationAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalOperations.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/operations', request_field='', request_type_name='ComputeGlobalOperationsAggregatedListRequest', response_type_name='OperationAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified Operations resource.

      Args:
        request: (ComputeGlobalOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeGlobalOperationsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.globalOperations.delete', ordered_params=['project', 'operation'], path_params=['operation', 'project'], query_params=[], relative_path='projects/{project}/global/operations/{operation}', request_field='', request_type_name='ComputeGlobalOperationsDeleteRequest', response_type_name='ComputeGlobalOperationsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified Operations resource.

      Args:
        request: (ComputeGlobalOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalOperations.get', ordered_params=['project', 'operation'], path_params=['operation', 'project'], query_params=[], relative_path='projects/{project}/global/operations/{operation}', request_field='', request_type_name='ComputeGlobalOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of Operation resources contained within the specified project.

      Args:
        request: (ComputeGlobalOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OperationList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalOperations.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/operations', request_field='', request_type_name='ComputeGlobalOperationsListRequest', response_type_name='OperationList', supports_download=False)

    def Wait(self, request, global_params=None):
        """Waits for the specified Operation resource to return as `DONE` or for the request to approach the 2 minute deadline, and retrieves the specified Operation resource. This method differs from the `GET` method in that it waits for no more than the default deadline (2 minutes) and then returns the current state of the operation, which might be `DONE` or still in progress. This method is called on a best-effort basis. Specifically: - In uncommon cases, when the server is overloaded, the request might return before the default deadline is reached, or might return after zero seconds. - If the default deadline is reached, there is no guarantee that the operation is actually done when the method returns. Be prepared to retry if the operation is not `DONE`. .

      Args:
        request: (ComputeGlobalOperationsWaitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Wait')
        return self._RunMethod(config, request, global_params=global_params)
    Wait.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalOperations.wait', ordered_params=['project', 'operation'], path_params=['operation', 'project'], query_params=[], relative_path='projects/{project}/global/operations/{operation}/wait', request_field='', request_type_name='ComputeGlobalOperationsWaitRequest', response_type_name='Operation', supports_download=False)