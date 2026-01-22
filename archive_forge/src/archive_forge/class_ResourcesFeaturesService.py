from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class ResourcesFeaturesService(base_api.BaseApiService):
    """Service class for the resources_features resource."""
    _NAME = u'resources_features'

    def __init__(self, client):
        super(AdminDirectoryV1.ResourcesFeaturesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a feature.

      Args:
        request: (DirectoryResourcesFeaturesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryResourcesFeaturesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.resources.features.delete', ordered_params=[u'customer', u'featureKey'], path_params=[u'customer', u'featureKey'], query_params=[], relative_path=u'customer/{customer}/resources/features/{featureKey}', request_field='', request_type_name=u'DirectoryResourcesFeaturesDeleteRequest', response_type_name=u'DirectoryResourcesFeaturesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a feature.

      Args:
        request: (DirectoryResourcesFeaturesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Feature) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resources.features.get', ordered_params=[u'customer', u'featureKey'], path_params=[u'customer', u'featureKey'], query_params=[], relative_path=u'customer/{customer}/resources/features/{featureKey}', request_field='', request_type_name=u'DirectoryResourcesFeaturesGetRequest', response_type_name=u'Feature', supports_download=False)

    def Insert(self, request, global_params=None):
        """Inserts a feature.

      Args:
        request: (DirectoryResourcesFeaturesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Feature) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.resources.features.insert', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/features', request_field=u'feature', request_type_name=u'DirectoryResourcesFeaturesInsertRequest', response_type_name=u'Feature', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of features for an account.

      Args:
        request: (DirectoryResourcesFeaturesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Features) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resources.features.list', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[u'maxResults', u'pageToken'], relative_path=u'customer/{customer}/resources/features', request_field='', request_type_name=u'DirectoryResourcesFeaturesListRequest', response_type_name=u'Features', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a feature.

      This method supports patch semantics.

      Args:
        request: (DirectoryResourcesFeaturesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Feature) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.resources.features.patch', ordered_params=[u'customer', u'featureKey'], path_params=[u'customer', u'featureKey'], query_params=[], relative_path=u'customer/{customer}/resources/features/{featureKey}', request_field=u'feature', request_type_name=u'DirectoryResourcesFeaturesPatchRequest', response_type_name=u'Feature', supports_download=False)

    def Rename(self, request, global_params=None):
        """Renames a feature.

      Args:
        request: (DirectoryResourcesFeaturesRenameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryResourcesFeaturesRenameResponse) The response message.
      """
        config = self.GetMethodConfig('Rename')
        return self._RunMethod(config, request, global_params=global_params)
    Rename.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.resources.features.rename', ordered_params=[u'customer', u'oldName'], path_params=[u'customer', u'oldName'], query_params=[], relative_path=u'customer/{customer}/resources/features/{oldName}/rename', request_field=u'featureRename', request_type_name=u'DirectoryResourcesFeaturesRenameRequest', response_type_name=u'DirectoryResourcesFeaturesRenameResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a feature.

      Args:
        request: (DirectoryResourcesFeaturesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Feature) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.resources.features.update', ordered_params=[u'customer', u'featureKey'], path_params=[u'customer', u'featureKey'], query_params=[], relative_path=u'customer/{customer}/resources/features/{featureKey}', request_field=u'feature', request_type_name=u'DirectoryResourcesFeaturesUpdateRequest', response_type_name=u'Feature', supports_download=False)