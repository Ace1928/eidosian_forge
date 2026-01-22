from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.ml.v1 import ml_v1_messages as messages
class ProjectsModelsVersionsService(base_api.BaseApiService):
    """Service class for the projects_models_versions resource."""
    _NAME = 'projects_models_versions'

    def __init__(self, client):
        super(MlV1.ProjectsModelsVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new version of a model from a trained TensorFlow model. If the version created in the cloud by this call is the first deployed version of the specified model, it will be made the default version of the model. When you add a version to a model that already has one or more versions, the default version does not automatically change. If you want a new version to be the default, you must call projects.models.versions.setDefault.

      Args:
        request: (MlProjectsModelsVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/models/{modelsId}/versions', http_method='POST', method_id='ml.projects.models.versions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/versions', request_field='googleCloudMlV1Version', request_type_name='MlProjectsModelsVersionsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a model version. Each model can have multiple versions deployed and in use at any given time. Use this method to remove a single version. Note: You cannot delete the version that is set as the default version of the model unless it is the only remaining version.

      Args:
        request: (MlProjectsModelsVersionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/models/{modelsId}/versions/{versionsId}', http_method='DELETE', method_id='ml.projects.models.versions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='MlProjectsModelsVersionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a model version. Models can have multiple versions. You can call projects.models.versions.list to get the same information that this method returns for all of the versions of a model.

      Args:
        request: (MlProjectsModelsVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudMlV1Version) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/models/{modelsId}/versions/{versionsId}', http_method='GET', method_id='ml.projects.models.versions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='MlProjectsModelsVersionsGetRequest', response_type_name='GoogleCloudMlV1Version', supports_download=False)

    def List(self, request, global_params=None):
        """Gets basic information about all the versions of a model. If you expect that a model has many versions, or if you need to handle only a limited number of results at a time, you can request that the list be retrieved in batches (called pages). If there are no versions that match the request parameters, the list request returns an empty response body: {}.

      Args:
        request: (MlProjectsModelsVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudMlV1ListVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/models/{modelsId}/versions', http_method='GET', method_id='ml.projects.models.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/versions', request_field='', request_type_name='MlProjectsModelsVersionsListRequest', response_type_name='GoogleCloudMlV1ListVersionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified Version resource. Currently the only update-able fields are `description`, `requestLoggingConfig`, `autoScaling.minNodes`, and `manualScaling.nodes`.

      Args:
        request: (MlProjectsModelsVersionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/models/{modelsId}/versions/{versionsId}', http_method='PATCH', method_id='ml.projects.models.versions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudMlV1Version', request_type_name='MlProjectsModelsVersionsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetDefault(self, request, global_params=None):
        """Designates a version to be the default for the model. The default version is used for prediction requests made against the model that don't specify a version. The first version to be created for a model is automatically set as the default. You must make any subsequent changes to the default version setting manually using this method.

      Args:
        request: (MlProjectsModelsVersionsSetDefaultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudMlV1Version) The response message.
      """
        config = self.GetMethodConfig('SetDefault')
        return self._RunMethod(config, request, global_params=global_params)
    SetDefault.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/models/{modelsId}/versions/{versionsId}:setDefault', http_method='POST', method_id='ml.projects.models.versions.setDefault', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setDefault', request_field='googleCloudMlV1SetDefaultVersionRequest', request_type_name='MlProjectsModelsVersionsSetDefaultRequest', response_type_name='GoogleCloudMlV1Version', supports_download=False)