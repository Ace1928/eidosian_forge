from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v1 import cloudresourcemanager_v1_messages as messages
class LiensService(base_api.BaseApiService):
    """Service class for the liens resource."""
    _NAME = 'liens'

    def __init__(self, client):
        super(CloudresourcemanagerV1.LiensService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a Lien which applies to the resource denoted by the `parent` field. Callers of this method will require permission on the `parent` resource. For example, applying to `projects/1234` requires permission `resourcemanager.projects.updateLiens`. NOTE: Some resources may limit the number of Liens which may be applied.

      Args:
        request: (Lien) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Lien) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudresourcemanager.liens.create', ordered_params=[], path_params=[], query_params=[], relative_path='v1/liens', request_field='<request>', request_type_name='Lien', response_type_name='Lien', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a Lien by `name`. Callers of this method will require permission on the `parent` resource. For example, a Lien with a `parent` of `projects/1234` requires permission `resourcemanager.projects.updateLiens`.

      Args:
        request: (CloudresourcemanagerLiensDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='cloudresourcemanager.liens.delete', ordered_params=['liensId'], path_params=['liensId'], query_params=[], relative_path='v1/liens/{liensId}', request_field='', request_type_name='CloudresourcemanagerLiensDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve a Lien by `name`. Callers of this method will require permission on the `parent` resource. For example, a Lien with a `parent` of `projects/1234` requires permission `resourcemanager.projects.get`.

      Args:
        request: (CloudresourcemanagerLiensGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Lien) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.liens.get', ordered_params=['liensId'], path_params=['liensId'], query_params=[], relative_path='v1/liens/{liensId}', request_field='', request_type_name='CloudresourcemanagerLiensGetRequest', response_type_name='Lien', supports_download=False)

    def List(self, request, global_params=None):
        """List all Liens applied to the `parent` resource. Callers of this method will require permission on the `parent` resource. For example, a Lien with a `parent` of `projects/1234` requires permission `resourcemanager.projects.get`.

      Args:
        request: (CloudresourcemanagerLiensListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLiensResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.liens.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v1/liens', request_field='', request_type_name='CloudresourcemanagerLiensListRequest', response_type_name='ListLiensResponse', supports_download=False)