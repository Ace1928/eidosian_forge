from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsKeyvaluemapsEntriesService(base_api.BaseApiService):
    """Service class for the organizations_keyvaluemaps_entries resource."""
    _NAME = 'organizations_keyvaluemaps_entries'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsKeyvaluemapsEntriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates key value entries in a key value map scoped to an organization, environment, or API proxy. **Note**: Supported for Apigee hybrid 1.8.x and higher.

      Args:
        request: (ApigeeOrganizationsKeyvaluemapsEntriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1KeyValueEntry) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/keyvaluemaps/{keyvaluemapsId}/entries', http_method='POST', method_id='apigee.organizations.keyvaluemaps.entries.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/entries', request_field='googleCloudApigeeV1KeyValueEntry', request_type_name='ApigeeOrganizationsKeyvaluemapsEntriesCreateRequest', response_type_name='GoogleCloudApigeeV1KeyValueEntry', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a key value entry from a key value map scoped to an organization, environment, or API proxy. **Notes:** * After you delete the key value entry, the policy consuming the entry will continue to function with its cached values for a few minutes. This is expected behavior. * Supported for Apigee hybrid 1.8.x and higher.

      Args:
        request: (ApigeeOrganizationsKeyvaluemapsEntriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1KeyValueEntry) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/keyvaluemaps/{keyvaluemapsId}/entries/{entriesId}', http_method='DELETE', method_id='apigee.organizations.keyvaluemaps.entries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsKeyvaluemapsEntriesDeleteRequest', response_type_name='GoogleCloudApigeeV1KeyValueEntry', supports_download=False)

    def Get(self, request, global_params=None):
        """Get the key value entry value for a key value map scoped to an organization, environment, or API proxy. **Note**: Supported for Apigee hybrid 1.8.x and higher.

      Args:
        request: (ApigeeOrganizationsKeyvaluemapsEntriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1KeyValueEntry) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/keyvaluemaps/{keyvaluemapsId}/entries/{entriesId}', http_method='GET', method_id='apigee.organizations.keyvaluemaps.entries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsKeyvaluemapsEntriesGetRequest', response_type_name='GoogleCloudApigeeV1KeyValueEntry', supports_download=False)

    def List(self, request, global_params=None):
        """Lists key value entries for key values maps scoped to an organization, environment, or API proxy. **Note**: Supported for Apigee hybrid 1.8.x and higher.

      Args:
        request: (ApigeeOrganizationsKeyvaluemapsEntriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListKeyValueEntriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/keyvaluemaps/{keyvaluemapsId}/entries', http_method='GET', method_id='apigee.organizations.keyvaluemaps.entries.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/entries', request_field='', request_type_name='ApigeeOrganizationsKeyvaluemapsEntriesListRequest', response_type_name='GoogleCloudApigeeV1ListKeyValueEntriesResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Update key value entry scoped to an organization, environment, or API proxy for an existing key.

      Args:
        request: (GoogleCloudApigeeV1KeyValueEntry) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1KeyValueEntry) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/keyvaluemaps/{keyvaluemapsId}/entries/{entriesId}', http_method='PUT', method_id='apigee.organizations.keyvaluemaps.entries.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1KeyValueEntry', response_type_name='GoogleCloudApigeeV1KeyValueEntry', supports_download=False)