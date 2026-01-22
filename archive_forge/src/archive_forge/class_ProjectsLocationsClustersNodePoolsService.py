from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
class ProjectsLocationsClustersNodePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_clusters_nodePools resource."""
    _NAME = 'projects_locations_clusters_nodePools'

    def __init__(self, client):
        super(ContainerV1.ProjectsLocationsClustersNodePoolsService, self).__init__(client)
        self._upload_configs = {}

    def CompleteUpgrade(self, request, global_params=None):
        """CompleteNodePoolUpgrade will signal an on-going node pool upgrade to complete.

      Args:
        request: (ContainerProjectsLocationsClustersNodePoolsCompleteUpgradeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('CompleteUpgrade')
        return self._RunMethod(config, request, global_params=global_params)
    CompleteUpgrade.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}:completeUpgrade', http_method='POST', method_id='container.projects.locations.clusters.nodePools.completeUpgrade', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:completeUpgrade', request_field='completeNodePoolUpgradeRequest', request_type_name='ContainerProjectsLocationsClustersNodePoolsCompleteUpgradeRequest', response_type_name='Empty', supports_download=False)

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
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools', http_method='POST', method_id='container.projects.locations.clusters.nodePools.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/nodePools', request_field='<request>', request_type_name='CreateNodePoolRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a node pool from a cluster.

      Args:
        request: (ContainerProjectsLocationsClustersNodePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}', http_method='DELETE', method_id='container.projects.locations.clusters.nodePools.delete', ordered_params=['name'], path_params=['name'], query_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], relative_path='v1/{+name}', request_field='', request_type_name='ContainerProjectsLocationsClustersNodePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the requested node pool.

      Args:
        request: (ContainerProjectsLocationsClustersNodePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}', http_method='GET', method_id='container.projects.locations.clusters.nodePools.get', ordered_params=['name'], path_params=['name'], query_params=['clusterId', 'nodePoolId', 'projectId', 'zone'], relative_path='v1/{+name}', request_field='', request_type_name='ContainerProjectsLocationsClustersNodePoolsGetRequest', response_type_name='NodePool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the node pools for a cluster.

      Args:
        request: (ContainerProjectsLocationsClustersNodePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNodePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools', http_method='GET', method_id='container.projects.locations.clusters.nodePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['clusterId', 'projectId', 'zone'], relative_path='v1/{+parent}/nodePools', request_field='', request_type_name='ContainerProjectsLocationsClustersNodePoolsListRequest', response_type_name='ListNodePoolsResponse', supports_download=False)

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
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}:rollback', http_method='POST', method_id='container.projects.locations.clusters.nodePools.rollback', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rollback', request_field='<request>', request_type_name='RollbackNodePoolUpgradeRequest', response_type_name='Operation', supports_download=False)

    def SetAutoscaling(self, request, global_params=None):
        """Sets the autoscaling settings for the specified node pool.

      Args:
        request: (SetNodePoolAutoscalingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetAutoscaling')
        return self._RunMethod(config, request, global_params=global_params)
    SetAutoscaling.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}:setAutoscaling', http_method='POST', method_id='container.projects.locations.clusters.nodePools.setAutoscaling', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setAutoscaling', request_field='<request>', request_type_name='SetNodePoolAutoscalingRequest', response_type_name='Operation', supports_download=False)

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
    SetManagement.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}:setManagement', http_method='POST', method_id='container.projects.locations.clusters.nodePools.setManagement', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setManagement', request_field='<request>', request_type_name='SetNodePoolManagementRequest', response_type_name='Operation', supports_download=False)

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
    SetSize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}:setSize', http_method='POST', method_id='container.projects.locations.clusters.nodePools.setSize', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setSize', request_field='<request>', request_type_name='SetNodePoolSizeRequest', response_type_name='Operation', supports_download=False)

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
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/nodePools/{nodePoolsId}', http_method='PUT', method_id='container.projects.locations.clusters.nodePools.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='UpdateNodePoolRequest', response_type_name='Operation', supports_download=False)