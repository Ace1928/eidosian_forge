from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
class ProjectsLocationsDatasetsAnnotationStoresAnnotationsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_annotationStores_annotations resource."""
    _NAME = 'projects_locations_datasets_annotationStores_annotations'

    def __init__(self, client):
        super(HealthcareV1beta1.ProjectsLocationsDatasetsAnnotationStoresAnnotationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Annotation record. It is valid to create Annotation objects for the same source more than once since a unique ID is assigned to each record by this service.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Annotation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}/annotations', http_method='POST', method_id='healthcare.projects.locations.datasets.annotationStores.annotations.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/annotations', request_field='annotation', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsCreateRequest', response_type_name='Annotation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Annotation or returns NOT_FOUND if it does not exist.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}/annotations/{annotationsId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.annotationStores.annotations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an Annotation.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Annotation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}/annotations/{annotationsId}', http_method='GET', method_id='healthcare.projects.locations.datasets.annotationStores.annotations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsGetRequest', response_type_name='Annotation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Annotations in the given Annotation store for a source resource.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAnnotationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}/annotations', http_method='GET', method_id='healthcare.projects.locations.datasets.annotationStores.annotations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'view'], relative_path='v1beta1/{+parent}/annotations', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsListRequest', response_type_name='ListAnnotationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the Annotation.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Annotation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}/annotations/{annotationsId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.annotationStores.annotations.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='annotation', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresAnnotationsPatchRequest', response_type_name='Annotation', supports_download=False)