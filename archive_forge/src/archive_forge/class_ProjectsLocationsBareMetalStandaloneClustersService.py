from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsBareMetalStandaloneClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_bareMetalStandaloneClusters resource."""
    _NAME = 'projects_locations_bareMetalStandaloneClusters'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsBareMetalStandaloneClustersService, self).__init__(client)
        self._upload_configs = {}

    def Enroll(self, request, global_params=None):
        """Enrolls an existing bare metal standalone cluster to the Anthos On-Prem API within a given project and location. Through enrollment, an existing standalone cluster will become Anthos On-Prem API managed. The corresponding GCP resources will be created and all future modifications to the cluster will be expected to be performed through the API.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersEnrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enroll')
        return self._RunMethod(config, request, global_params=global_params)
    Enroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters:enroll', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.enroll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/bareMetalStandaloneClusters:enroll', request_field='enrollBareMetalStandaloneClusterRequest', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersEnrollRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single bare metal standalone cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BareMetalStandaloneCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersGetRequest', response_type_name='BareMetalStandaloneCluster', supports_download=False)

    def List(self, request, global_params=None):
        """Lists bare metal standalone clusters in a given project and location.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBareMetalStandaloneClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/bareMetalStandaloneClusters', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersListRequest', response_type_name='ListBareMetalStandaloneClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single bare metal standalone cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}', http_method='PATCH', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='bareMetalStandaloneCluster', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersPatchRequest', response_type_name='Operation', supports_download=False)

    def QueryVersionConfig(self, request, global_params=None):
        """Queries the bare metal standalone cluster version config.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersQueryVersionConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryBareMetalStandaloneVersionConfigResponse) The response message.
      """
        config = self.GetMethodConfig('QueryVersionConfig')
        return self._RunMethod(config, request, global_params=global_params)
    QueryVersionConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters:queryVersionConfig', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.queryVersionConfig', ordered_params=['parent'], path_params=['parent'], query_params=['upgradeConfig_clusterName'], relative_path='v1/{+parent}/bareMetalStandaloneClusters:queryVersionConfig', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersQueryVersionConfigRequest', response_type_name='QueryBareMetalStandaloneVersionConfigResponse', supports_download=False)

    def Unenroll(self, request, global_params=None):
        """Unenrolls an existing bare metal standalone cluster from the GKE on-prem API within a given project and location. Unenrollment removes the Cloud reference to the cluster without modifying the underlying OnPrem Resources. Clusters will continue to run; however, they will no longer be accessible through the Anthos On-Prem API or its clients.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalStandaloneClustersUnenrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unenroll')
        return self._RunMethod(config, request, global_params=global_params)
    Unenroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalStandaloneClusters/{bareMetalStandaloneClustersId}:unenroll', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalStandaloneClusters.unenroll', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'force', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}:unenroll', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalStandaloneClustersUnenrollRequest', response_type_name='Operation', supports_download=False)