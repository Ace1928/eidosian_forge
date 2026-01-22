from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsBareMetalClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_bareMetalClusters resource."""
    _NAME = 'projects_locations_bareMetalClusters'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsBareMetalClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new bare metal cluster in a given project and location.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['bareMetalClusterId', 'validateOnly'], relative_path='v1/{+parent}/bareMetalClusters', request_field='bareMetalCluster', request_type_name='GkeonpremProjectsLocationsBareMetalClustersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single bare metal Cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalClusters.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'force', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Enroll(self, request, global_params=None):
        """Enrolls an existing bare metal user cluster and its node pools to the Anthos On-Prem API within a given project and location. Through enrollment, an existing cluster will become Anthos On-Prem API managed. The corresponding GCP resources will be created and all future modifications to the cluster and/or its node pools will be expected to be performed through the API.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersEnrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enroll')
        return self._RunMethod(config, request, global_params=global_params)
    Enroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters:enroll', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.enroll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/bareMetalClusters:enroll', request_field='enrollBareMetalClusterRequest', request_type_name='GkeonpremProjectsLocationsBareMetalClustersEnrollRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single bare metal Cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BareMetalCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.get', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'view'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersGetRequest', response_type_name='BareMetalCluster', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}:getIamPolicy', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists bare metal clusters in a given project and location.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBareMetalClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['allowMissing', 'filter', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/bareMetalClusters', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersListRequest', response_type_name='ListBareMetalClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single bare metal Cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}', http_method='PATCH', method_id='gkeonprem.projects.locations.bareMetalClusters.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='bareMetalCluster', request_type_name='GkeonpremProjectsLocationsBareMetalClustersPatchRequest', response_type_name='Operation', supports_download=False)

    def QueryVersionConfig(self, request, global_params=None):
        """Queries the bare metal user cluster version config.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersQueryVersionConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryBareMetalVersionConfigResponse) The response message.
      """
        config = self.GetMethodConfig('QueryVersionConfig')
        return self._RunMethod(config, request, global_params=global_params)
    QueryVersionConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters:queryVersionConfig', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.queryVersionConfig', ordered_params=['parent'], path_params=['parent'], query_params=['createConfig_adminClusterMembership', 'createConfig_adminClusterName', 'upgradeConfig_clusterName'], relative_path='v1/{+parent}/bareMetalClusters:queryVersionConfig', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersQueryVersionConfigRequest', response_type_name='QueryBareMetalVersionConfigResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}:setIamPolicy', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkeonpremProjectsLocationsBareMetalClustersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}:testIamPermissions', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkeonpremProjectsLocationsBareMetalClustersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Unenroll(self, request, global_params=None):
        """Unenrolls an existing bare metal user cluster and its node pools from the Anthos On-Prem API within a given project and location. Unenrollment removes the Cloud reference to the cluster without modifying the underlying OnPrem Resources. Clusters and node pools will continue to run; however, they will no longer be accessible through the Anthos On-Prem API or its clients.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersUnenrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unenroll')
        return self._RunMethod(config, request, global_params=global_params)
    Unenroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}:unenroll', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalClusters.unenroll', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'force', 'validateOnly'], relative_path='v1/{+name}:unenroll', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersUnenrollRequest', response_type_name='Operation', supports_download=False)