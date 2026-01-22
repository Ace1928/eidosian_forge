from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ZoneOperationsService(base_api.BaseApiService):
    """Service class for the zoneOperations resource."""
    _NAME = 'zoneOperations'

    def __init__(self, client):
        super(ComputeBeta.ZoneOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified zone-specific Operations resource.

      Args:
        request: (ComputeZoneOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeZoneOperationsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.zoneOperations.delete', ordered_params=['project', 'zone', 'operation'], path_params=['operation', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/operations/{operation}', request_field='', request_type_name='ComputeZoneOperationsDeleteRequest', response_type_name='ComputeZoneOperationsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified zone-specific Operations resource.

      Args:
        request: (ComputeZoneOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.zoneOperations.get', ordered_params=['project', 'zone', 'operation'], path_params=['operation', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/operations/{operation}', request_field='', request_type_name='ComputeZoneOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of Operation resources contained within the specified zone.

      Args:
        request: (ComputeZoneOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OperationList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.zoneOperations.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/operations', request_field='', request_type_name='ComputeZoneOperationsListRequest', response_type_name='OperationList', supports_download=False)

    def Wait(self, request, global_params=None):
        """Waits for the specified Operation resource to return as `DONE` or for the request to approach the 2 minute deadline, and retrieves the specified Operation resource. This method waits for no more than the 2 minutes and then returns the current state of the operation, which might be `DONE` or still in progress. This method is called on a best-effort basis. Specifically: - In uncommon cases, when the server is overloaded, the request might return before the default deadline is reached, or might return after zero seconds. - If the default deadline is reached, there is no guarantee that the operation is actually done when the method returns. Be prepared to retry if the operation is not `DONE`. .

      Args:
        request: (ComputeZoneOperationsWaitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Wait')
        return self._RunMethod(config, request, global_params=global_params)
    Wait.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.zoneOperations.wait', ordered_params=['project', 'zone', 'operation'], path_params=['operation', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/operations/{operation}/wait', request_field='', request_type_name='ComputeZoneOperationsWaitRequest', response_type_name='Operation', supports_download=False)