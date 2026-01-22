from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
class ProjectsZonesClustersService(base_api.BaseApiService):
    """Service class for the projects_zones_clusters resource."""
    _NAME = 'projects_zones_clusters'

    def __init__(self, client):
        super(ContainerV1.ProjectsZonesClustersService, self).__init__(client)
        self._upload_configs = {}

    def Addons(self, request, global_params=None):
        """Sets the addons for a specific cluster.

      Args:
        request: (SetAddonsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Addons')
        return self._RunMethod(config, request, global_params=global_params)
    Addons.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.addons', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/addons', request_field='<request>', request_type_name='SetAddonsConfigRequest', response_type_name='Operation', supports_download=False)

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
    CompleteIpRotation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.completeIpRotation', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}:completeIpRotation', request_field='<request>', request_type_name='CompleteIPRotationRequest', response_type_name='Operation', supports_download=False)

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
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.create', ordered_params=['projectId', 'zone'], path_params=['projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters', request_field='<request>', request_type_name='CreateClusterRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the cluster, including the Kubernetes endpoint and all worker nodes. Firewalls and routes that were configured during cluster creation are also deleted. Other Google Compute Engine resources that might be in use by the cluster, such as load balancer resources, are not deleted if they weren't present when the cluster was initially created.

      Args:
        request: (ContainerProjectsZonesClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='container.projects.zones.clusters.delete', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=['name'], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}', request_field='', request_type_name='ContainerProjectsZonesClustersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a specific cluster.

      Args:
        request: (ContainerProjectsZonesClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Cluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='container.projects.zones.clusters.get', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=['name'], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}', request_field='', request_type_name='ContainerProjectsZonesClustersGetRequest', response_type_name='Cluster', supports_download=False)

    def LegacyAbac(self, request, global_params=None):
        """Enables or disables the ABAC authorization mechanism on a cluster.

      Args:
        request: (SetLegacyAbacRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('LegacyAbac')
        return self._RunMethod(config, request, global_params=global_params)
    LegacyAbac.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.legacyAbac', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/legacyAbac', request_field='<request>', request_type_name='SetLegacyAbacRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all clusters owned by a project in either the specified zone or all zones.

      Args:
        request: (ContainerProjectsZonesClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='container.projects.zones.clusters.list', ordered_params=['projectId', 'zone'], path_params=['projectId', 'zone'], query_params=['parent'], relative_path='v1/projects/{projectId}/zones/{zone}/clusters', request_field='', request_type_name='ContainerProjectsZonesClustersListRequest', response_type_name='ListClustersResponse', supports_download=False)

    def Locations(self, request, global_params=None):
        """Sets the locations for a specific cluster. Deprecated. Use [projects.locations.clusters.update](https://cloud.google.com/kubernetes-engine/docs/reference/rest/v1/projects.locations.clusters/update) instead.

      Args:
        request: (SetLocationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Locations')
        return self._RunMethod(config, request, global_params=global_params)
    Locations.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.locations', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/locations', request_field='<request>', request_type_name='SetLocationsRequest', response_type_name='Operation', supports_download=False)

    def Logging(self, request, global_params=None):
        """Sets the logging service for a specific cluster.

      Args:
        request: (SetLoggingServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Logging')
        return self._RunMethod(config, request, global_params=global_params)
    Logging.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.logging', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/logging', request_field='<request>', request_type_name='SetLoggingServiceRequest', response_type_name='Operation', supports_download=False)

    def Master(self, request, global_params=None):
        """Updates the master for a specific cluster.

      Args:
        request: (UpdateMasterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Master')
        return self._RunMethod(config, request, global_params=global_params)
    Master.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.master', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/master', request_field='<request>', request_type_name='UpdateMasterRequest', response_type_name='Operation', supports_download=False)

    def Monitoring(self, request, global_params=None):
        """Sets the monitoring service for a specific cluster.

      Args:
        request: (SetMonitoringServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Monitoring')
        return self._RunMethod(config, request, global_params=global_params)
    Monitoring.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.monitoring', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/monitoring', request_field='<request>', request_type_name='SetMonitoringServiceRequest', response_type_name='Operation', supports_download=False)

    def ResourceLabels(self, request, global_params=None):
        """Sets labels on a cluster.

      Args:
        request: (SetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResourceLabels')
        return self._RunMethod(config, request, global_params=global_params)
    ResourceLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.resourceLabels', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}/resourceLabels', request_field='<request>', request_type_name='SetLabelsRequest', response_type_name='Operation', supports_download=False)

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
    SetMaintenancePolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.setMaintenancePolicy', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}:setMaintenancePolicy', request_field='<request>', request_type_name='SetMaintenancePolicyRequest', response_type_name='Operation', supports_download=False)

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
    SetMasterAuth.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.setMasterAuth', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}:setMasterAuth', request_field='<request>', request_type_name='SetMasterAuthRequest', response_type_name='Operation', supports_download=False)

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
    SetNetworkPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.setNetworkPolicy', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}:setNetworkPolicy', request_field='<request>', request_type_name='SetNetworkPolicyRequest', response_type_name='Operation', supports_download=False)

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
    StartIpRotation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.clusters.startIpRotation', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}:startIpRotation', request_field='<request>', request_type_name='StartIPRotationRequest', response_type_name='Operation', supports_download=False)

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
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='container.projects.zones.clusters.update', ordered_params=['projectId', 'zone', 'clusterId'], path_params=['clusterId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/clusters/{clusterId}', request_field='<request>', request_type_name='UpdateClusterRequest', response_type_name='Operation', supports_download=False)