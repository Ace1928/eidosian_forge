from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3 import translate_v3_messages as messages
class ProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesService(base_api.BaseApiService):
    """Service class for the projects_locations_adaptiveMtDatasets_adaptiveMtFiles resource."""
    _NAME = 'projects_locations_adaptiveMtDatasets_adaptiveMtFiles'

    def __init__(self, client):
        super(TranslateV3.ProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes an AdaptiveMtFile along with its sentences.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets/{adaptiveMtDatasetsId}/adaptiveMtFiles/{adaptiveMtFilesId}', http_method='DELETE', method_id='translate.projects.locations.adaptiveMtDatasets.adaptiveMtFiles.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets and AdaptiveMtFile.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AdaptiveMtFile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets/{adaptiveMtDatasetsId}/adaptiveMtFiles/{adaptiveMtFilesId}', http_method='GET', method_id='translate.projects.locations.adaptiveMtDatasets.adaptiveMtFiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesGetRequest', response_type_name='AdaptiveMtFile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all AdaptiveMtFiles associated to an AdaptiveMtDataset.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAdaptiveMtFilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/locations/{locationsId}/adaptiveMtDatasets/{adaptiveMtDatasetsId}/adaptiveMtFiles', http_method='GET', method_id='translate.projects.locations.adaptiveMtDatasets.adaptiveMtFiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v3/{+parent}/adaptiveMtFiles', request_field='', request_type_name='TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesListRequest', response_type_name='ListAdaptiveMtFilesResponse', supports_download=False)