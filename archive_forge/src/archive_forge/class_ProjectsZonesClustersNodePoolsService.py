from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
class ProjectsZonesClustersNodePoolsService(base_api.BaseApiService):
    """Service class for the projects_zones_clusters_nodePools resource."""
    _NAME = 'projects_zones_clusters_nodePools'

    def __init__(self, client):
        super(ContainerV1.ProjectsZonesClustersNodePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Autoscaling(self, request, global_params=None):
        """Sets the autoscaling settings for the specified node pool.

      Args:
        request: (SetNodePoolAutoscalingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Autoscaling')
        return self._RunMethod(config, request, global_params=global_params)
    Autoscaling.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.nodePools.autoscaling', ordered_params=['projectId', 'zone', 'clusterId', 'nodePoolId'], path_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}/autoscaling', request_field='<request>', request_type_name='SetNodePoolAutoscalingRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a node pool for a cluster.

      Args:
        request: (CreateNodePoolRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.nodePools.create', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools', request_field='<request>', request_type_name='CreateNodePoolRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a node pool from a cluster.

      Args:
        request: (ContainerProjectsZonesClustersNodePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='container.projects.zones.clusters.nodePools.delete', ordered_params=['projectId', 'zone', 'clusterId', 'nodePoolId'], path_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], query_params=['name'], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}', request_field='', request_type_name='ContainerProjectsZonesClustersNodePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the requested node pool.

      Args:
        request: (ContainerProjectsZonesClustersNodePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='container.projects.zones.clusters.nodePools.get', ordered_params=['projectId', 'zone', 'clusterId', 'nodePoolId'], path_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], query_params=['name'], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}', request_field='', request_type_name='ContainerProjectsZonesClustersNodePoolsGetRequest', response_type_name='NodePool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the node pools for a cluster.

      Args:
        request: (ContainerProjectsZonesClustersNodePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNodePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='container.projects.zones.clusters.nodePools.list', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=['parent'], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools', request_field='', request_type_name='ContainerProjectsZonesClustersNodePoolsListRequest', response_type_name='ListNodePoolsResponse', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Rolls back a previously Aborted or Failed NodePool upgrade. This makes no changes if the last upgrade successfully completed.

      Args:
        request: (RollbackNodePoolUpgradeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.nodePools.rollback', ordered_params=['projectId', 'zone', 'clusterId', 'nodePoolId'], path_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}:rollback', request_field='<request>', request_type_name='RollbackNodePoolUpgradeRequest', response_type_name='Operation', supports_download=False)

    def SetManagement(self, request, global_params=None):
        """Sets the NodeManagement options for a node pool.

      Args:
        request: (SetNodePoolManagementRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetManagement')
        return self._RunMethod(config, request, global_params=global_params)
    SetManagement.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.nodePools.setManagement', ordered_params=['projectId', 'zone', 'clusterId', 'nodePoolId'], path_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}/setManagement', request_field='<request>', request_type_name='SetNodePoolManagementRequest', response_type_name='Operation', supports_download=False)

    def SetSize(self, request, global_params=None):
        """Sets the size for a specific node pool. The new size will be used for all replicas, including future replicas created by modifying NodePool.locations.

      Args:
        request: (SetNodePoolSizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetSize')
        return self._RunMethod(config, request, global_params=global_params)
    SetSize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.nodePools.setSize', ordered_params=['projectId', 'zone', 'clusterId', 'nodePoolId'], path_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}/setSize', request_field='<request>', request_type_name='SetNodePoolSizeRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the version and/or image type for the specified node pool.

      Args:
        request: (UpdateNodePoolRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.nodePools.update', ordered_params=['projectId', 'zone', 'clusterId', 'nodePoolId'], path_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/nodePools/{nodePoolId}/update', request_field='<request>', request_type_name='UpdateNodePoolRequest', response_type_name='Operation', supports_download=False)