from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
class ProjectsLocationsAwsClustersAwsNodePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_awsClusters_awsNodePools resource."""
    _NAME = 'projects_locations_awsClusters_awsNodePools'

    def __init__(self, client):
        super(GkemulticloudV1.ProjectsLocationsAwsClustersAwsNodePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AwsNodePool, attached to a given AwsCluster. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/awsNodePools', http_method='POST', method_id='gkemulticloud.projects.locations.awsClusters.awsNodePools.create', ordered_params=['parent'], path_params=['parent'], query_params=['awsNodePoolId', 'validateOnly'], relative_path='v1/{+parent}/awsNodePools', request_field='googleCloudGkemulticloudV1AwsNodePool', request_type_name='GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific AwsNodePool resource. If successful, the response contains a newly created Operation resource that can be described to track the status of the operation.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/awsNodePools/{awsNodePoolsId}', http_method='DELETE', method_id='gkemulticloud.projects.locations.awsClusters.awsNodePools.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Describes a specific AwsNodePool resource.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AwsNodePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/awsNodePools/{awsNodePoolsId}', http_method='GET', method_id='gkemulticloud.projects.locations.awsClusters.awsNodePools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsGetRequest', response_type_name='GoogleCloudGkemulticloudV1AwsNodePool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all AwsNodePool resources on a given AwsCluster.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1ListAwsNodePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/awsNodePools', http_method='GET', method_id='gkemulticloud.projects.locations.awsClusters.awsNodePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/awsNodePools', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsListRequest', response_type_name='GoogleCloudGkemulticloudV1ListAwsNodePoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an AwsNodePool.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/awsNodePools/{awsNodePoolsId}', http_method='PATCH', method_id='gkemulticloud.projects.locations.awsClusters.awsNodePools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='googleCloudGkemulticloudV1AwsNodePool', request_type_name='GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Rolls back a previously aborted or failed AwsNodePool update request. Makes no changes if the last update request successfully finished. If an update request is in progress, you cannot rollback the update. You must first cancel or let it finish unsuccessfully before you can rollback.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/awsNodePools/{awsNodePoolsId}:rollback', http_method='POST', method_id='gkemulticloud.projects.locations.awsClusters.awsNodePools.rollback', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rollback', request_field='googleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest', request_type_name='GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsRollbackRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)