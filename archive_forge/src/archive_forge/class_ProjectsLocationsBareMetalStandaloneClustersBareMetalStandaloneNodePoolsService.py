from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_bareMetalStandaloneClusters_bareMetalStandaloneNodePools resource."""
    _NAME = 'projects_locations_bareMetalStandaloneClusters_bareMetalStandaloneNodePools'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new bare metal standalone node pool in a given project, location and bare metal standalone cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}/bareMetalStandaloneNodePools', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.bareMetalStandaloneNodePools.create', ordered_params=['parent'], path_params=['parent'], query_params=['bareMetalStandaloneNodePoolId', 'validateOnly'], relative_path='v1/{+parent}/bareMetalStandaloneNodePools', request_field='bareMetalStandaloneNodePool', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single bare metal standalone node pool.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}/bareMetalStandaloneNodePools/{bareMetalStandaloneNodePoolsId}', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.bareMetalStandaloneNodePools.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Enroll(self, request, global_params=None):
        """Enrolls an existing bare metal standalone node pool to the Anthos On-Prem API within a given project and location. Through enrollment, an existing standalone node pool will become Anthos On-Prem API managed. The corresponding GCP resources will be created.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsEnrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enroll')
        return self._RunMethod(config, request, global_params=global_params)
    Enroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}/bareMetalStandaloneNodePools:enroll', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.bareMetalStandaloneNodePools.enroll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/bareMetalStandaloneNodePools:enroll', request_field='enrollBareMetalStandaloneNodePoolRequest', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsEnrollRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single bare metal standalone node pool.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BareMetalStandaloneNodePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}/bareMetalStandaloneNodePools/{bareMetalStandaloneNodePoolsId}', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.bareMetalStandaloneNodePools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsGetRequest', response_type_name='BareMetalStandaloneNodePool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists bare metal standalone node pools in a given project, location and bare metal standalone cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBareMetalStandaloneNodePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}/bareMetalStandaloneNodePools', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.bareMetalStandaloneNodePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/bareMetalStandaloneNodePools', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsListRequest', response_type_name='ListBareMetalStandaloneNodePoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single bare metal standalone node pool.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}/bareMetalStandaloneNodePools/{bareMetalStandaloneNodePoolsId}', http_method='PATCH', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.bareMetalStandaloneNodePools.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='bareMetalStandaloneNodePool', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsPatchRequest', response_type_name='Operation', supports_download=False)

    def Unenroll(self, request, global_params=None):
        """Unenrolls a bare metal standalone node pool from Anthos On-Prem API.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsUnenrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unenroll')
        return self._RunMethod(config, request, global_params=global_params)
    Unenroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}/bareMetalStandaloneNodePools/{bareMetalStandaloneNodePoolsId}:unenroll', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.bareMetalStandaloneNodePools.unenroll', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v1/{+name}:unenroll', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsUnenrollRequest', response_type_name='Operation', supports_download=False)