from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsRegionsClustersNodeGroupsService(base_api.BaseApiService):
    """Service class for the projects_regions_clusters_nodeGroups resource."""
    _NAME = 'projects_regions_clusters_nodeGroups'

    def __init__(self, client):
        super(DataprocV1.ProjectsRegionsClustersNodeGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a node group in a cluster. The returned Operation.metadata is NodeGroupOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#nodegroupoperationmetadata).

      Args:
        request: (DataprocProjectsRegionsClustersNodeGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}/nodeGroups', http_method='POST', method_id='dataproc.projects.regions.clusters.nodeGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['nodeGroupId', 'parentOperationId', 'requestId'], relative_path='v1/{+parent}/nodeGroups', request_field='nodeGroup', request_type_name='DataprocProjectsRegionsClustersNodeGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a node group in a cluster. The returned Operation.metadata is NodeGroupOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#nodegroupoperationmetadata).

      Args:
        request: (DataprocProjectsRegionsClustersNodeGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}/nodeGroups/{nodeGroupsId}', http_method='DELETE', method_id='dataproc.projects.regions.clusters.nodeGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsRegionsClustersNodeGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the resource representation for a node group in a cluster.

      Args:
        request: (DataprocProjectsRegionsClustersNodeGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}/nodeGroups/{nodeGroupsId}', http_method='GET', method_id='dataproc.projects.regions.clusters.nodeGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsRegionsClustersNodeGroupsGetRequest', response_type_name='NodeGroup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all node groups in a cluster.

      Args:
        request: (DataprocProjectsRegionsClustersNodeGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNodeGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}/nodeGroups', http_method='GET', method_id='dataproc.projects.regions.clusters.nodeGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/nodeGroups', request_field='', request_type_name='DataprocProjectsRegionsClustersNodeGroupsListRequest', response_type_name='ListNodeGroupsResponse', supports_download=False)

    def Repair(self, request, global_params=None):
        """Repair nodes in a node group.

      Args:
        request: (DataprocProjectsRegionsClustersNodeGroupsRepairRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Repair')
        return self._RunMethod(config, request, global_params=global_params)
    Repair.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}/nodeGroups/{nodeGroupsId}:repair', http_method='POST', method_id='dataproc.projects.regions.clusters.nodeGroups.repair', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:repair', request_field='repairNodeGroupRequest', request_type_name='DataprocProjectsRegionsClustersNodeGroupsRepairRequest', response_type_name='Operation', supports_download=False)

    def Resize(self, request, global_params=None):
        """Resizes a node group in a cluster. The returned Operation.metadata is NodeGroupOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#nodegroupoperationmetadata).

      Args:
        request: (DataprocProjectsRegionsClustersNodeGroupsResizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resize')
        return self._RunMethod(config, request, global_params=global_params)
    Resize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}/nodeGroups/{nodeGroupsId}:resize', http_method='POST', method_id='dataproc.projects.regions.clusters.nodeGroups.resize', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resize', request_field='resizeNodeGroupRequest', request_type_name='DataprocProjectsRegionsClustersNodeGroupsResizeRequest', response_type_name='Operation', supports_download=False)

    def UpdateLabels(self, request, global_params=None):
        """Updates labels on the node group in a cluster. The returned Operation.metadata is NodeGroupOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#nodegroupoperationmetadata).

      Args:
        request: (DataprocProjectsRegionsClustersNodeGroupsUpdateLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateLabels')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateLabels.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}/nodeGroups/{nodeGroupsId}:updateLabels', http_method='POST', method_id='dataproc.projects.regions.clusters.nodeGroups.updateLabels', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:updateLabels', request_field='updateLabelsNodeGroupRequest', request_type_name='DataprocProjectsRegionsClustersNodeGroupsUpdateLabelsRequest', response_type_name='Operation', supports_download=False)