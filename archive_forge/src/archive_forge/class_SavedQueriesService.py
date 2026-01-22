from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
class SavedQueriesService(base_api.BaseApiService):
    """Service class for the savedQueries resource."""
    _NAME = 'savedQueries'

    def __init__(self, client):
        super(CloudassetV1.SavedQueriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a saved query in a parent project/folder/organization.

      Args:
        request: (CloudassetSavedQueriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SavedQuery) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/savedQueries', http_method='POST', method_id='cloudasset.savedQueries.create', ordered_params=['parent'], path_params=['parent'], query_params=['savedQueryId'], relative_path='v1/{+parent}/savedQueries', request_field='savedQuery', request_type_name='CloudassetSavedQueriesCreateRequest', response_type_name='SavedQuery', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a saved query.

      Args:
        request: (CloudassetSavedQueriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/savedQueries/{savedQueriesId}', http_method='DELETE', method_id='cloudasset.savedQueries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudassetSavedQueriesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a saved query.

      Args:
        request: (CloudassetSavedQueriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SavedQuery) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/savedQueries/{savedQueriesId}', http_method='GET', method_id='cloudasset.savedQueries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudassetSavedQueriesGetRequest', response_type_name='SavedQuery', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all saved queries in a parent project/folder/organization.

      Args:
        request: (CloudassetSavedQueriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSavedQueriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/savedQueries', http_method='GET', method_id='cloudasset.savedQueries.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/savedQueries', request_field='', request_type_name='CloudassetSavedQueriesListRequest', response_type_name='ListSavedQueriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a saved query.

      Args:
        request: (CloudassetSavedQueriesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SavedQuery) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/{v1Id}/{v1Id1}/savedQueries/{savedQueriesId}', http_method='PATCH', method_id='cloudasset.savedQueries.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='savedQuery', request_type_name='CloudassetSavedQueriesPatchRequest', response_type_name='SavedQuery', supports_download=False)