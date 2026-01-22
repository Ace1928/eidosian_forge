from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataprocgdc.v1alpha1 import dataprocgdc_v1alpha1_messages as messages
class ProjectsLocationsServiceInstancesSparkApplicationsService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceInstances_sparkApplications resource."""
    _NAME = 'projects_locations_serviceInstances_sparkApplications'

    def __init__(self, client):
        super(DataprocgdcV1alpha1.ProjectsLocationsServiceInstancesSparkApplicationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an application associated with a Dataproc operator.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/sparkApplications', http_method='POST', method_id='dataprocgdc.projects.locations.serviceInstances.sparkApplications.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'sparkApplicationId'], relative_path='v1alpha1/{+parent}/sparkApplications', request_field='sparkApplication', request_type_name='DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a application.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/sparkApplications/{sparkApplicationsId}', http_method='DELETE', method_id='dataprocgdc.projects.locations.serviceInstances.sparkApplications.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a application.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SparkApplication) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/sparkApplications/{sparkApplicationsId}', http_method='GET', method_id='dataprocgdc.projects.locations.serviceInstances.sparkApplications.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsGetRequest', response_type_name='SparkApplication', supports_download=False)

    def List(self, request, global_params=None):
        """Lists applications in a location.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSparkApplicationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/sparkApplications', http_method='GET', method_id='dataprocgdc.projects.locations.serviceInstances.sparkApplications.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/sparkApplications', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsListRequest', response_type_name='ListSparkApplicationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a application. Only supports updating state or labels.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/sparkApplications/{sparkApplicationsId}', http_method='PATCH', method_id='dataprocgdc.projects.locations.serviceInstances.sparkApplications.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='sparkApplication', request_type_name='DataprocgdcProjectsLocationsServiceInstancesSparkApplicationsPatchRequest', response_type_name='Operation', supports_download=False)