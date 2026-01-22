from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storagetransfer.v1 import storagetransfer_v1_messages as messages
class TransferOperationsService(base_api.BaseApiService):
    """Service class for the transferOperations resource."""
    _NAME = 'transferOperations'

    def __init__(self, client):
        super(StoragetransferV1.TransferOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a transfer. Use the transferOperations.get method to check if the cancellation succeeded or if the operation completed despite the `cancel` request. When you cancel an operation, the currently running transfer is interrupted. For recurring transfer jobs, the next instance of the transfer job will still run. For example, if your job is configured to run every day at 1pm and you cancel Monday's operation at 1:05pm, Monday's transfer will stop. However, a transfer job will still be attempted on Tuesday. This applies only to currently running operations. If an operation is not currently running, `cancel` does nothing. *Caution:* Canceling a transfer job can leave your data in an unknown state. We recommend that you restore the state at both the destination and the source after the `cancel` request completes so that your data is in a consistent state. When you cancel a job, the next job computes a delta of files and may repair any inconsistent state. For instance, if you run a job every day, and today's job found 10 new files and transferred five files before you canceled the job, tomorrow's transfer operation will compute a new delta with the five files that were not copied today plus any new files discovered tomorrow.

      Args:
        request: (StoragetransferTransferOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferOperations/{transferOperationsId}:cancel', http_method='POST', method_id='storagetransfer.transferOperations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelOperationRequest', request_type_name='StoragetransferTransferOperationsCancelRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (StoragetransferTransferOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferOperations/{transferOperationsId}', http_method='GET', method_id='storagetransfer.transferOperations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='StoragetransferTransferOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists transfer operations. Operations are ordered by their creation time in reverse chronological order.

      Args:
        request: (StoragetransferTransferOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferOperations', http_method='GET', method_id='storagetransfer.transferOperations.list', ordered_params=['name', 'filter'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+name}', request_field='', request_type_name='StoragetransferTransferOperationsListRequest', response_type_name='ListOperationsResponse', supports_download=False)

    def Pause(self, request, global_params=None):
        """Pauses a transfer operation.

      Args:
        request: (StoragetransferTransferOperationsPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferOperations/{transferOperationsId}:pause', http_method='POST', method_id='storagetransfer.transferOperations.pause', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:pause', request_field='pauseTransferOperationRequest', request_type_name='StoragetransferTransferOperationsPauseRequest', response_type_name='Empty', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resumes a transfer operation that is paused.

      Args:
        request: (StoragetransferTransferOperationsResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/transferOperations/{transferOperationsId}:resume', http_method='POST', method_id='storagetransfer.transferOperations.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resume', request_field='resumeTransferOperationRequest', request_type_name='StoragetransferTransferOperationsResumeRequest', response_type_name='Empty', supports_download=False)