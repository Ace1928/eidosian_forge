from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class OrganizationsLocationsSavedQueriesService(base_api.BaseApiService):
    """Service class for the organizations_locations_savedQueries resource."""
    _NAME = 'organizations_locations_savedQueries'

    def __init__(self, client):
        super(LoggingV2.OrganizationsLocationsSavedQueriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new SavedQuery for the user making the request.

      Args:
        request: (LoggingOrganizationsLocationsSavedQueriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SavedQuery) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/savedQueries', http_method='POST', method_id='logging.organizations.locations.savedQueries.create', ordered_params=['parent'], path_params=['parent'], query_params=['savedQueryId'], relative_path='v2/{+parent}/savedQueries', request_field='savedQuery', request_type_name='LoggingOrganizationsLocationsSavedQueriesCreateRequest', response_type_name='SavedQuery', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing SavedQuery that was created by the user making the request.

      Args:
        request: (LoggingOrganizationsLocationsSavedQueriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/savedQueries/{savedQueriesId}', http_method='DELETE', method_id='logging.organizations.locations.savedQueries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingOrganizationsLocationsSavedQueriesDeleteRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the SavedQueries that were created by the user making the request.

      Args:
        request: (LoggingOrganizationsLocationsSavedQueriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSavedQueriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/savedQueries', http_method='GET', method_id='logging.organizations.locations.savedQueries.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/savedQueries', request_field='', request_type_name='LoggingOrganizationsLocationsSavedQueriesListRequest', response_type_name='ListSavedQueriesResponse', supports_download=False)