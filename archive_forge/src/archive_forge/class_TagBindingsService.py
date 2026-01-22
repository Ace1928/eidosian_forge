from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v3 import cloudresourcemanager_v3_messages as messages
class TagBindingsService(base_api.BaseApiService):
    """Service class for the tagBindings resource."""
    _NAME = 'tagBindings'

    def __init__(self, client):
        super(CloudresourcemanagerV3.TagBindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a TagBinding between a TagValue and a Google Cloud resource.

      Args:
        request: (CloudresourcemanagerTagBindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudresourcemanager.tagBindings.create', ordered_params=[], path_params=[], query_params=['validateOnly'], relative_path='v3/tagBindings', request_field='tagBinding', request_type_name='CloudresourcemanagerTagBindingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TagBinding.

      Args:
        request: (CloudresourcemanagerTagBindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagBindings/{tagBindingsId}', http_method='DELETE', method_id='cloudresourcemanager.tagBindings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='CloudresourcemanagerTagBindingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the TagBindings for the given Google Cloud resource, as specified with `parent`. NOTE: The `parent` field is expected to be a full resource name: https://cloud.google.com/apis/design/resource_names#full_resource_name.

      Args:
        request: (CloudresourcemanagerTagBindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTagBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.tagBindings.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v3/tagBindings', request_field='', request_type_name='CloudresourcemanagerTagBindingsListRequest', response_type_name='ListTagBindingsResponse', supports_download=False)