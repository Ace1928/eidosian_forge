from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
class OtherCloudConnectionsService(base_api.BaseApiService):
    """Service class for the otherCloudConnections resource."""
    _NAME = 'otherCloudConnections'

    def __init__(self, client):
        super(CloudassetV1.OtherCloudConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an other-cloud connection under a parent scope.

      Args:
        request: (CloudassetOtherCloudConnectionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OtherCloudConnection) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/otherCloudConnections', http_method='POST', method_id='cloudasset.otherCloudConnections.create', ordered_params=['parent'], path_params=['parent'], query_params=['otherCloudConnectionId'], relative_path='v1/{+parent}/otherCloudConnections', request_field='otherCloudConnection', request_type_name='CloudassetOtherCloudConnectionsCreateRequest', response_type_name='OtherCloudConnection', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an other-cloud connection.

      Args:
        request: (CloudassetOtherCloudConnectionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/otherCloudConnections/{otherCloudConnectionsId}', http_method='DELETE', method_id='cloudasset.otherCloudConnections.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudassetOtherCloudConnectionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an other-cloud connection detail.

      Args:
        request: (CloudassetOtherCloudConnectionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OtherCloudConnection) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/otherCloudConnections/{otherCloudConnectionsId}', http_method='GET', method_id='cloudasset.otherCloudConnections.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudassetOtherCloudConnectionsGetRequest', response_type_name='OtherCloudConnection', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all other-cloud connections under a parent scope.

      Args:
        request: (CloudassetOtherCloudConnectionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOtherCloudConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/otherCloudConnections', http_method='GET', method_id='cloudasset.otherCloudConnections.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/otherCloudConnections', request_field='', request_type_name='CloudassetOtherCloudConnectionsListRequest', response_type_name='ListOtherCloudConnectionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an other-cloud connection under a parent scope.

      Args:
        request: (CloudassetOtherCloudConnectionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OtherCloudConnection) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/otherCloudConnections/{otherCloudConnectionsId}', http_method='PATCH', method_id='cloudasset.otherCloudConnections.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='otherCloudConnection', request_type_name='CloudassetOtherCloudConnectionsPatchRequest', response_type_name='OtherCloudConnection', supports_download=False)

    def Verify(self, request, global_params=None):
        """Verifies the validity of an other-cloud connection, and writes the validation result into spanner if the connection exists. A connection will be considered as valid if the GCP service account can be assumed to the AWS delegated role successfully.

      Args:
        request: (VerifyOtherCloudConnectionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VerifyOtherCloudConnectionResponse) The response message.
      """
        config = self.GetMethodConfig('Verify')
        return self._RunMethod(config, request, global_params=global_params)
    Verify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/otherCloudConnections/{otherCloudConnectionsId}:verify', http_method='POST', method_id='cloudasset.otherCloudConnections.verify', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:verify', request_field='<request>', request_type_name='VerifyOtherCloudConnectionRequest', response_type_name='VerifyOtherCloudConnectionResponse', supports_download=False)