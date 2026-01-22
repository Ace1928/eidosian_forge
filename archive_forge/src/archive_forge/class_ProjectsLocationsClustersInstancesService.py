from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.alloydb.v1beta import alloydb_v1beta_messages as messages
class ProjectsLocationsClustersInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_clusters_instances resource."""
    _NAME = 'projects_locations_clusters_instances'

    def __init__(self, client):
        super(AlloydbV1beta.ProjectsLocationsClustersInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Instance in a given project and location.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances', http_method='POST', method_id='alloydb.projects.locations.clusters.instances.create', ordered_params=['parent'], path_params=['parent'], query_params=['instanceId', 'requestId', 'validateOnly'], relative_path='v1beta/{+parent}/instances', request_field='instance', request_type_name='AlloydbProjectsLocationsClustersInstancesCreateRequest', response_type_name='Operation', supports_download=False)

    def Createsecondary(self, request, global_params=None):
        """Creates a new SECONDARY Instance in a given project and location.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesCreatesecondaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Createsecondary')
        return self._RunMethod(config, request, global_params=global_params)
    Createsecondary.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances:createsecondary', http_method='POST', method_id='alloydb.projects.locations.clusters.instances.createsecondary', ordered_params=['parent'], path_params=['parent'], query_params=['instanceId', 'requestId', 'validateOnly'], relative_path='v1beta/{+parent}/instances:createsecondary', request_field='instance', request_type_name='AlloydbProjectsLocationsClustersInstancesCreatesecondaryRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Instance.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances/{instancesId}', http_method='DELETE', method_id='alloydb.projects.locations.clusters.instances.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId', 'validateOnly'], relative_path='v1beta/{+name}', request_field='', request_type_name='AlloydbProjectsLocationsClustersInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Failover(self, request, global_params=None):
        """Forces a Failover for a highly available instance. Failover promotes the HA standby instance as the new primary. Imperative only.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesFailoverRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Failover')
        return self._RunMethod(config, request, global_params=global_params)
    Failover.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances/{instancesId}:failover', http_method='POST', method_id='alloydb.projects.locations.clusters.instances.failover', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:failover', request_field='failoverInstanceRequest', request_type_name='AlloydbProjectsLocationsClustersInstancesFailoverRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Instance.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Instance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances/{instancesId}', http_method='GET', method_id='alloydb.projects.locations.clusters.instances.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1beta/{+name}', request_field='', request_type_name='AlloydbProjectsLocationsClustersInstancesGetRequest', response_type_name='Instance', supports_download=False)

    def GetConnectionInfo(self, request, global_params=None):
        """Get instance metadata used for a connection.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesGetConnectionInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectionInfo) The response message.
      """
        config = self.GetMethodConfig('GetConnectionInfo')
        return self._RunMethod(config, request, global_params=global_params)
    GetConnectionInfo.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances/{instancesId}/connectionInfo', http_method='GET', method_id='alloydb.projects.locations.clusters.instances.getConnectionInfo', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1beta/{+parent}/connectionInfo', request_field='', request_type_name='AlloydbProjectsLocationsClustersInstancesGetConnectionInfoRequest', response_type_name='ConnectionInfo', supports_download=False)

    def InjectFault(self, request, global_params=None):
        """Injects fault in an instance. Imperative only.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesInjectFaultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('InjectFault')
        return self._RunMethod(config, request, global_params=global_params)
    InjectFault.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances/{instancesId}:injectFault', http_method='POST', method_id='alloydb.projects.locations.clusters.instances.injectFault', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:injectFault', request_field='injectFaultRequest', request_type_name='AlloydbProjectsLocationsClustersInstancesInjectFaultRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Instances in a given project and location.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances', http_method='GET', method_id='alloydb.projects.locations.clusters.instances.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/instances', request_field='', request_type_name='AlloydbProjectsLocationsClustersInstancesListRequest', response_type_name='ListInstancesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Instance.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances/{instancesId}', http_method='PATCH', method_id='alloydb.projects.locations.clusters.instances.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1beta/{+name}', request_field='instance', request_type_name='AlloydbProjectsLocationsClustersInstancesPatchRequest', response_type_name='Operation', supports_download=False)

    def Restart(self, request, global_params=None):
        """Restart an Instance in a cluster. Imperative only.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesRestartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Restart')
        return self._RunMethod(config, request, global_params=global_params)
    Restart.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/instances/{instancesId}:restart', http_method='POST', method_id='alloydb.projects.locations.clusters.instances.restart', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:restart', request_field='restartInstanceRequest', request_type_name='AlloydbProjectsLocationsClustersInstancesRestartRequest', response_type_name='Operation', supports_download=False)