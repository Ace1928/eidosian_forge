from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsBareMetalAdminClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_bareMetalAdminClusters resource."""
    _NAME = 'projects_locations_bareMetalAdminClusters'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsBareMetalAdminClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new bare metal admin cluster in a given project and location. The API needs to be combined with creating a bootstrap cluster to work. See: https://cloud.google.com/anthos/clusters/docs/bare-metal/latest/installing/creating-clusters/create-admin-cluster-api#prepare_bootstrap_environment.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['bareMetalAdminClusterId', 'validateOnly'], relative_path='v1/{+parent}/bareMetalAdminClusters', request_field='bareMetalAdminCluster', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersCreateRequest', response_type_name='Operation', supports_download=False)

    def Enroll(self, request, global_params=None):
        """Enrolls an existing bare metal admin cluster to the Anthos On-Prem API within a given project and location. Through enrollment, an existing admin cluster will become Anthos On-Prem API managed. The corresponding GCP resources will be created and all future modifications to the cluster will be expected to be performed through the API.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersEnrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enroll')
        return self._RunMethod(config, request, global_params=global_params)
    Enroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters:enroll', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.enroll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/bareMetalAdminClusters:enroll', request_field='enrollBareMetalAdminClusterRequest', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersEnrollRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single bare metal admin cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BareMetalAdminCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters/{bareMetalAdminClustersId}', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.get', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'view'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersGetRequest', response_type_name='BareMetalAdminCluster', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters/{bareMetalAdminClustersId}:getIamPolicy', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists bare metal admin clusters in a given project and location.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBareMetalAdminClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['allowMissing', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/bareMetalAdminClusters', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersListRequest', response_type_name='ListBareMetalAdminClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single bare metal admin cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters/{bareMetalAdminClustersId}', http_method='PATCH', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='bareMetalAdminCluster', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersPatchRequest', response_type_name='Operation', supports_download=False)

    def QueryVersionConfig(self, request, global_params=None):
        """Queries the bare metal admin cluster version config.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersQueryVersionConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryBareMetalAdminVersionConfigResponse) The response message.
      """
        config = self.GetMethodConfig('QueryVersionConfig')
        return self._RunMethod(config, request, global_params=global_params)
    QueryVersionConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters:queryVersionConfig', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.queryVersionConfig', ordered_params=['parent'], path_params=['parent'], query_params=['upgradeConfig_clusterName'], relative_path='v1/{+parent}/bareMetalAdminClusters:queryVersionConfig', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersQueryVersionConfigRequest', response_type_name='QueryBareMetalAdminVersionConfigResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters/{bareMetalAdminClustersId}:setIamPolicy', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters/{bareMetalAdminClustersId}:testIamPermissions', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Unenroll(self, request, global_params=None):
        """Unenrolls an existing bare metal admin cluster from the Anthos On-Prem API within a given project and location. Unenrollment removes the Cloud reference to the cluster without modifying the underlying OnPrem Resources. Clusters will continue to run; however, they will no longer be accessible through the Anthos On-Prem API or its clients.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalAdminClustersUnenrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unenroll')
        return self._RunMethod(config, request, global_params=global_params)
    Unenroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalAdminClusters/{bareMetalAdminClustersId}:unenroll', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalAdminClusters.unenroll', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}:unenroll', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalAdminClustersUnenrollRequest', response_type_name='Operation', supports_download=False)