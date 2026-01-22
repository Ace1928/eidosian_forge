from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
class ProjectsLocationsApplicationsWorkloadsService(base_api.BaseApiService):
    """Service class for the projects_locations_applications_workloads resource."""
    _NAME = 'projects_locations_applications_workloads'

    def __init__(self, client):
        super(ApphubV1alpha.ProjectsLocationsApplicationsWorkloadsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Workload in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsWorkloadsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/workloads', http_method='POST', method_id='apphub.projects.locations.applications.workloads.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'workloadId'], relative_path='v1alpha/{+parent}/workloads', request_field='workload', request_type_name='ApphubProjectsLocationsApplicationsWorkloadsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Workload from an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsWorkloadsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/workloads/{workloadsId}', http_method='DELETE', method_id='apphub.projects.locations.applications.workloads.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsApplicationsWorkloadsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Workload in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsWorkloadsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workload) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/workloads/{workloadsId}', http_method='GET', method_id='apphub.projects.locations.applications.workloads.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsApplicationsWorkloadsGetRequest', response_type_name='Workload', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Workloads in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsWorkloadsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/workloads', http_method='GET', method_id='apphub.projects.locations.applications.workloads.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/workloads', request_field='', request_type_name='ApphubProjectsLocationsApplicationsWorkloadsListRequest', response_type_name='ListWorkloadsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Workload in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsWorkloadsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/workloads/{workloadsId}', http_method='PATCH', method_id='apphub.projects.locations.applications.workloads.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='workload', request_type_name='ApphubProjectsLocationsApplicationsWorkloadsPatchRequest', response_type_name='Operation', supports_download=False)