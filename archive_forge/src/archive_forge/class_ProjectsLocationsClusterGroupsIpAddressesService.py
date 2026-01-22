from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sddc.v1alpha1 import sddc_v1alpha1_messages as messages
class ProjectsLocationsClusterGroupsIpAddressesService(base_api.BaseApiService):
    """Service class for the projects_locations_clusterGroups_ipAddresses resource."""
    _NAME = 'projects_locations_clusterGroups_ipAddresses'

    def __init__(self, client):
        super(SddcV1alpha1.ProjectsLocationsClusterGroupsIpAddressesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `IpAddress` in a given `ClusterGroup`. The creation is asynchronous. You can check the returned operation to track its progress. When the operation successfully completes, the cluster is fully functional. The returned operation is automatically deleted after a few hours, so there is no need to call `DeleteOperation`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsIpAddressesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/ipAddresses', http_method='POST', method_id='sddc.projects.locations.clusterGroups.ipAddresses.create', ordered_params=['parent'], path_params=['parent'], query_params=['ipAddressId'], relative_path='v1alpha1/{+parent}/ipAddresses', request_field='ipAddress', request_type_name='SddcProjectsLocationsClusterGroupsIpAddressesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an `IpAddress` in a given `ClusterGroup`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsIpAddressesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/ipAddresses/{ipAddressesId}', http_method='DELETE', method_id='sddc.projects.locations.clusterGroups.ipAddresses.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsIpAddressesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a single `IpAddress`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsIpAddressesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IpAddress) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/ipAddresses/{ipAddressesId}', http_method='GET', method_id='sddc.projects.locations.clusterGroups.ipAddresses.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsIpAddressesGetRequest', response_type_name='IpAddress', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `IpAddress` objects in a given `ClusterGroup`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsIpAddressesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListIpAddressesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}/ipAddresses', http_method='GET', method_id='sddc.projects.locations.clusterGroups.ipAddresses.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/ipAddresses', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsIpAddressesListRequest', response_type_name='ListIpAddressesResponse', supports_download=False)