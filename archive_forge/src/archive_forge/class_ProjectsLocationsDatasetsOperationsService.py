from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsDatasetsOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_operations resource."""
    _NAME = 'projects_locations_datasets_operations'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsDatasetsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.

      Args:
        request: (AiplatformProjectsLocationsDatasetsOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/operations/{operationsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.datasets.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:cancel', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsOperationsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.

      Args:
        request: (AiplatformProjectsLocationsDatasetsOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/operations/{operationsId}', http_method='DELETE', method_id='aiplatform.projects.locations.datasets.operations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsOperationsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (AiplatformProjectsLocationsDatasetsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/operations/{operationsId}', http_method='GET', method_id='aiplatform.projects.locations.datasets.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (AiplatformProjectsLocationsDatasetsOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/operations', http_method='GET', method_id='aiplatform.projects.locations.datasets.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta1/{+name}/operations', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsOperationsListRequest', response_type_name='GoogleLongrunningListOperationsResponse', supports_download=False)

    def Wait(self, request, global_params=None):
        """Waits until the specified long-running operation is done or reaches at most a specified timeout, returning the latest state. If the operation is already done, the latest state is immediately returned. If the timeout specified is greater than the default HTTP/RPC timeout, the HTTP/RPC timeout is used. If the server does not support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Note that this method is on a best-effort basis. It may return the latest state before the specified timeout (including immediately), meaning even an immediate response is no guarantee that the operation is done.

      Args:
        request: (AiplatformProjectsLocationsDatasetsOperationsWaitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Wait')
        return self._RunMethod(config, request, global_params=global_params)
    Wait.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/operations/{operationsId}:wait', http_method='POST', method_id='aiplatform.projects.locations.datasets.operations.wait', ordered_params=['name'], path_params=['name'], query_params=['timeout'], relative_path='v1beta1/{+name}:wait', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsOperationsWaitRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)