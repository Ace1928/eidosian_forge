from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
class ProjectsLocationsClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_clusters resource."""
    _NAME = 'projects_locations_clusters'

    def __init__(self, client):
        super(ContainerV1.ProjectsLocationsClustersService, self).__init__(client)
        self._upload_configs = {}

    def CheckAutopilotCompatibility(self, request, global_params=None):
        """Checks the cluster compatibility with Autopilot mode, and returns a list of compatibility issues.

      Args:
        request: (ContainerProjectsLocationsClustersCheckAutopilotCompatibilityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckAutopilotCompatibilityResponse) The response message.
      """
        config = self.GetMethodConfig('CheckAutopilotCompatibility')
        return self._RunMethod(config, request, global_params=global_params)
    CheckAutopilotCompatibility.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:checkAutopilotCompatibility', http_method='GET', method_id='container.projects.locations.clusters.checkAutopilotCompatibility', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:checkAutopilotCompatibility', request_field='', request_type_name='ContainerProjectsLocationsClustersCheckAutopilotCompatibilityRequest', response_type_name='CheckAutopilotCompatibilityResponse', supports_download=False)

    def CompleteConvertToAutopilot(self, request, global_params=None):
        """CompleteConvertToAutopilot is an optional API that commits the conversion by deleting all Standard node pools and completing CA rotation. This action requires that a conversion has been started and that workload migration has completed, with no pods running on GKE Standard node pools. This action will be automatically performed 72 hours after conversion.

      Args:
        request: (ContainerProjectsLocationsClustersCompleteConvertToAutopilotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CompleteConvertToAutopilot')
        return self._RunMethod(config, request, global_params=global_params)
    CompleteConvertToAutopilot.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:CompleteConvertToAutopilot', http_method='POST', method_id='container.projects.locations.clusters.completeConvertToAutopilot', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:CompleteConvertToAutopilot', request_field='completeConvertToAutopilotRequest', request_type_name='ContainerProjectsLocationsClustersCompleteConvertToAutopilotRequest', response_type_name='Operation', supports_download=False)

    def CompleteIpRotation(self, request, global_params=None):
        """Completes master IP rotation.

      Args:
        request: (CompleteIPRotationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CompleteIpRotation')
        return self._RunMethod(config, request, global_params=global_params)
    CompleteIpRotation.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:completeIpRotation', http_method='POST', method_id='container.projects.locations.clusters.completeIpRotation', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:completeIpRotation', request_field='<request>', request_type_name='CompleteIPRotationRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a cluster, consisting of the specified number and type of Google Compute Engine instances. By default, the cluster is created in the project's [default network](https://cloud.google.com/compute/docs/networks-and-firewalls#networks). One firewall is added for the cluster. After cluster creation, the Kubelet creates routes for each node to allow the containers on that node to communicate with all other instances in the cluster. Finally, an entry is added to the project's global metadata indicating which CIDR range the cluster is using.

      Args:
        request: (CreateClusterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters', http_method='POST', method_id='container.projects.locations.clusters.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/clusters', request_field='<request>', request_type_name='CreateClusterRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the cluster, including the Kubernetes endpoint and all worker nodes. Firewalls and routes that were configured during cluster creation are also deleted. Other Google Compute Engine resources that might be in use by the cluster, such as load balancer resources, are not deleted if they weren't present when the cluster was initially created.

      Args:
        request: (ContainerProjectsLocationsClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}', http_method='DELETE', method_id='container.projects.locations.clusters.delete', ordered_params=['name'], path_params=['name'], query_params=['clusterId', 'projectId', 'zone'], relative_path='v1/{+name}', request_field='', request_type_name='ContainerProjectsLocationsClustersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a specific cluster.

      Args:
        request: (ContainerProjectsLocationsClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Cluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}', http_method='GET', method_id='container.projects.locations.clusters.get', ordered_params=['name'], path_params=['name'], query_params=['clusterId', 'projectId', 'zone'], relative_path='v1/{+name}', request_field='', request_type_name='ContainerProjectsLocationsClustersGetRequest', response_type_name='Cluster', supports_download=False)

    def GetJwks(self, request, global_params=None):
        """Gets the public component of the cluster signing keys in JSON Web Key format.

      Args:
        request: (ContainerProjectsLocationsClustersGetJwksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetJSONWebKeysResponse) The response message.
      """
        config = self.GetMethodConfig('GetJwks')
        return self._RunMethod(config, request, global_params=global_params)
    GetJwks.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/jwks', http_method='GET', method_id='container.projects.locations.clusters.getJwks', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/jwks', request_field='', request_type_name='ContainerProjectsLocationsClustersGetJwksRequest', response_type_name='GetJSONWebKeysResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all clusters owned by a project in either the specified zone or all zones.

      Args:
        request: (ContainerProjectsLocationsClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters', http_method='GET', method_id='container.projects.locations.clusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['projectId', 'zone'], relative_path='v1/{+parent}/clusters', request_field='', request_type_name='ContainerProjectsLocationsClustersListRequest', response_type_name='ListClustersResponse', supports_download=False)

    def SetAddons(self, request, global_params=None):
        """Sets the addons for a specific cluster.

      Args:
        request: (SetAddonsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetAddons')
        return self._RunMethod(config, request, global_params=global_params)
    SetAddons.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setAddons', http_method='POST', method_id='container.projects.locations.clusters.setAddons', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setAddons', request_field='<request>', request_type_name='SetAddonsConfigRequest', response_type_name='Operation', supports_download=False)

    def SetLegacyAbac(self, request, global_params=None):
        """Enables or disables the ABAC authorization mechanism on a cluster.

      Args:
        request: (SetLegacyAbacRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLegacyAbac')
        return self._RunMethod(config, request, global_params=global_params)
    SetLegacyAbac.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setLegacyAbac', http_method='POST', method_id='container.projects.locations.clusters.setLegacyAbac', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setLegacyAbac', request_field='<request>', request_type_name='SetLegacyAbacRequest', response_type_name='Operation', supports_download=False)

    def SetLocations(self, request, global_params=None):
        """Sets the locations for a specific cluster. Deprecated. Use [projects.locations.clusters.update](https://cloud.google.com/kubernetes-engine/docs/reference/rest/v1/projects.locations.clusters/update) instead.

      Args:
        request: (SetLocationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLocations')
        return self._RunMethod(config, request, global_params=global_params)
    SetLocations.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setLocations', http_method='POST', method_id='container.projects.locations.clusters.setLocations', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setLocations', request_field='<request>', request_type_name='SetLocationsRequest', response_type_name='Operation', supports_download=False)

    def SetLogging(self, request, global_params=None):
        """Sets the logging service for a specific cluster.

      Args:
        request: (SetLoggingServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLogging')
        return self._RunMethod(config, request, global_params=global_params)
    SetLogging.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setLogging', http_method='POST', method_id='container.projects.locations.clusters.setLogging', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setLogging', request_field='<request>', request_type_name='SetLoggingServiceRequest', response_type_name='Operation', supports_download=False)

    def SetMaintenancePolicy(self, request, global_params=None):
        """Sets the maintenance policy for a cluster.

      Args:
        request: (SetMaintenancePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetMaintenancePolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetMaintenancePolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setMaintenancePolicy', http_method='POST', method_id='container.projects.locations.clusters.setMaintenancePolicy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setMaintenancePolicy', request_field='<request>', request_type_name='SetMaintenancePolicyRequest', response_type_name='Operation', supports_download=False)

    def SetMasterAuth(self, request, global_params=None):
        """Sets master auth materials. Currently supports changing the admin password or a specific cluster, either via password generation or explicitly setting the password.

      Args:
        request: (SetMasterAuthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetMasterAuth')
        return self._RunMethod(config, request, global_params=global_params)
    SetMasterAuth.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setMasterAuth', http_method='POST', method_id='container.projects.locations.clusters.setMasterAuth', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setMasterAuth', request_field='<request>', request_type_name='SetMasterAuthRequest', response_type_name='Operation', supports_download=False)

    def SetMonitoring(self, request, global_params=None):
        """Sets the monitoring service for a specific cluster.

      Args:
        request: (SetMonitoringServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetMonitoring')
        return self._RunMethod(config, request, global_params=global_params)
    SetMonitoring.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setMonitoring', http_method='POST', method_id='container.projects.locations.clusters.setMonitoring', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setMonitoring', request_field='<request>', request_type_name='SetMonitoringServiceRequest', response_type_name='Operation', supports_download=False)

    def SetNetworkPolicy(self, request, global_params=None):
        """Enables or disables Network Policy for a cluster.

      Args:
        request: (SetNetworkPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetNetworkPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetNetworkPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setNetworkPolicy', http_method='POST', method_id='container.projects.locations.clusters.setNetworkPolicy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setNetworkPolicy', request_field='<request>', request_type_name='SetNetworkPolicyRequest', response_type_name='Operation', supports_download=False)

    def SetResourceLabels(self, request, global_params=None):
        """Sets labels on a cluster.

      Args:
        request: (SetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetResourceLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetResourceLabels.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:setResourceLabels', http_method='POST', method_id='container.projects.locations.clusters.setResourceLabels', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:setResourceLabels', request_field='<request>', request_type_name='SetLabelsRequest', response_type_name='Operation', supports_download=False)

    def StartIpRotation(self, request, global_params=None):
        """Starts master IP rotation.

      Args:
        request: (StartIPRotationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StartIpRotation')
        return self._RunMethod(config, request, global_params=global_params)
    StartIpRotation.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:startIpRotation', http_method='POST', method_id='container.projects.locations.clusters.startIpRotation', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:startIpRotation', request_field='<request>', request_type_name='StartIPRotationRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the settings of a specific cluster.

      Args:
        request: (UpdateClusterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}', http_method='PUT', method_id='container.projects.locations.clusters.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='UpdateClusterRequest', response_type_name='Operation', supports_download=False)

    def UpdateMaster(self, request, global_params=None):
        """Updates the master for a specific cluster.

      Args:
        request: (UpdateMasterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateMaster')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateMaster.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}:updateMaster', http_method='POST', method_id='container.projects.locations.clusters.updateMaster', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:updateMaster', request_field='<request>', request_type_name='UpdateMasterRequest', response_type_name='Operation', supports_download=False)