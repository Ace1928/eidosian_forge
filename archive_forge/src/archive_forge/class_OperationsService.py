from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
class OperationsService(base_api.BaseApiService):
    """Service class for the operations resource."""
    _NAME = u'operations'

    def __init__(self, client):
        super(ServicemanagementV1.OperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation.  Clients can use this.
method to poll the operation result at intervals as recommended by the API
service.

      Args:
        request: (ServicemanagementOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'servicemanagement.operations.get', ordered_params=[u'operationsId'], path_params=[u'operationsId'], query_params=[], relative_path=u'v1/operations/{operationsId}', request_field='', request_type_name=u'ServicemanagementOperationsGetRequest', response_type_name=u'Operation', supports_download=False)