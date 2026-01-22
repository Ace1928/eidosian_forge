from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.deploymentmanager.alpha import deploymentmanager_alpha_messages as messages
class CompositeTypesService(base_api.BaseApiService):
    """Service class for the compositeTypes resource."""
    _NAME = 'compositeTypes'

    def __init__(self, client):
        super(DeploymentmanagerAlpha.CompositeTypesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a composite type.

      Args:
        request: (DeploymentmanagerCompositeTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='deploymentmanager.compositeTypes.delete', ordered_params=['project', 'compositeType'], path_params=['compositeType', 'project'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/compositeTypes/{compositeType}', request_field='', request_type_name='DeploymentmanagerCompositeTypesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a specific composite type.

      Args:
        request: (DeploymentmanagerCompositeTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CompositeType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.compositeTypes.get', ordered_params=['project', 'compositeType'], path_params=['compositeType', 'project'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/compositeTypes/{compositeType}', request_field='', request_type_name='DeploymentmanagerCompositeTypesGetRequest', response_type_name='CompositeType', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a composite type.

      Args:
        request: (DeploymentmanagerCompositeTypesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='deploymentmanager.compositeTypes.insert', ordered_params=['project'], path_params=['project'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/compositeTypes', request_field='compositeType', request_type_name='DeploymentmanagerCompositeTypesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all composite types for Deployment Manager.

      Args:
        request: (DeploymentmanagerCompositeTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CompositeTypesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='deploymentmanager.compositeTypes.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken'], relative_path='deploymentmanager/alpha/projects/{project}/global/compositeTypes', request_field='', request_type_name='DeploymentmanagerCompositeTypesListRequest', response_type_name='CompositeTypesListResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches a composite type.

      Args:
        request: (DeploymentmanagerCompositeTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='deploymentmanager.compositeTypes.patch', ordered_params=['project', 'compositeType'], path_params=['compositeType', 'project'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/compositeTypes/{compositeType}', request_field='compositeTypeResource', request_type_name='DeploymentmanagerCompositeTypesPatchRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a composite type.

      Args:
        request: (DeploymentmanagerCompositeTypesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='deploymentmanager.compositeTypes.update', ordered_params=['project', 'compositeType'], path_params=['compositeType', 'project'], query_params=[], relative_path='deploymentmanager/alpha/projects/{project}/global/compositeTypes/{compositeType}', request_field='compositeTypeResource', request_type_name='DeploymentmanagerCompositeTypesUpdateRequest', response_type_name='Operation', supports_download=False)