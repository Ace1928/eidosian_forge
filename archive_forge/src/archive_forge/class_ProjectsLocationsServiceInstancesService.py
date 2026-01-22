from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataprocgdc.v1alpha1 import dataprocgdc_v1alpha1_messages as messages
class ProjectsLocationsServiceInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceInstances resource."""
    _NAME = 'projects_locations_serviceInstances'

    def __init__(self, client):
        super(DataprocgdcV1alpha1.ProjectsLocationsServiceInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a service instance in a GDC cluster.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances', http_method='POST', method_id='dataprocgdc.projects.locations.serviceInstances.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'serviceInstanceId'], relative_path='v1alpha1/{+parent}/serviceInstances', request_field='serviceInstance', request_type_name='DataprocgdcProjectsLocationsServiceInstancesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a service instance. Deleting will remove the service instance from the cluster, and deletes all Dataproc API objects from that cluster.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}', http_method='DELETE', method_id='dataprocgdc.projects.locations.serviceInstances.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'force', 'requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a service instance.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceInstance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}', http_method='GET', method_id='dataprocgdc.projects.locations.serviceInstances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesGetRequest', response_type_name='ServiceInstance', supports_download=False)

    def List(self, request, global_params=None):
        """Lists serviceInstances in a location.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances', http_method='GET', method_id='dataprocgdc.projects.locations.serviceInstances.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/serviceInstances', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesListRequest', response_type_name='ListServiceInstancesResponse', supports_download=False)