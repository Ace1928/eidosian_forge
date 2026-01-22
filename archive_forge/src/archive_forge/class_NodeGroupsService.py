from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NodeGroupsService(base_api.BaseApiService):
    """Service class for the nodeGroups resource."""
    _NAME = 'nodeGroups'

    def __init__(self, client):
        super(ComputeBeta.NodeGroupsService, self).__init__(client)
        self._upload_configs = {}

    def AddNodes(self, request, global_params=None):
        """Adds specified number of nodes to the node group.

      Args:
        request: (ComputeNodeGroupsAddNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddNodes')
        return self._RunMethod(config, request, global_params=global_params)
    AddNodes.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.addNodes', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}/addNodes', request_field='nodeGroupsAddNodesRequest', request_type_name='ComputeNodeGroupsAddNodesRequest', response_type_name='Operation', supports_download=False)

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of node groups. Note: use nodeGroups.listNodes for more details about each group. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeNodeGroupsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeGroupAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeGroups.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/nodeGroups', request_field='', request_type_name='ComputeNodeGroupsAggregatedListRequest', response_type_name='NodeGroupAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified NodeGroup resource.

      Args:
        request: (ComputeNodeGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.nodeGroups.delete', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}', request_field='', request_type_name='ComputeNodeGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def DeleteNodes(self, request, global_params=None):
        """Deletes specified nodes from the node group.

      Args:
        request: (ComputeNodeGroupsDeleteNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeleteNodes')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteNodes.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.deleteNodes', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}/deleteNodes', request_field='nodeGroupsDeleteNodesRequest', request_type_name='ComputeNodeGroupsDeleteNodesRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified NodeGroup. Get a list of available NodeGroups by making a list() request. Note: the "nodes" field should not be used. Use nodeGroups.listNodes instead.

      Args:
        request: (ComputeNodeGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeGroups.get', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}', request_field='', request_type_name='ComputeNodeGroupsGetRequest', response_type_name='NodeGroup', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeNodeGroupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeGroups.getIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{resource}/getIamPolicy', request_field='', request_type_name='ComputeNodeGroupsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a NodeGroup resource in the specified project using the data included in the request.

      Args:
        request: (ComputeNodeGroupsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.insert', ordered_params=['project', 'zone', 'initialNodeCount'], path_params=['project', 'zone'], query_params=['initialNodeCount', 'requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups', request_field='nodeGroup', request_type_name='ComputeNodeGroupsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of node groups available to the specified project. Note: use nodeGroups.listNodes for more details about each group.

      Args:
        request: (ComputeNodeGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeGroupList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeGroups.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/nodeGroups', request_field='', request_type_name='ComputeNodeGroupsListRequest', response_type_name='NodeGroupList', supports_download=False)

    def ListNodes(self, request, global_params=None):
        """Lists nodes in the node group.

      Args:
        request: (ComputeNodeGroupsListNodesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeGroupsListNodes) The response message.
      """
        config = self.GetMethodConfig('ListNodes')
        return self._RunMethod(config, request, global_params=global_params)
    ListNodes.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.listNodes', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}/listNodes', request_field='', request_type_name='ComputeNodeGroupsListNodesRequest', response_type_name='NodeGroupsListNodes', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified node group.

      Args:
        request: (ComputeNodeGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.nodeGroups.patch', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}', request_field='nodeGroupResource', request_type_name='ComputeNodeGroupsPatchRequest', response_type_name='Operation', supports_download=False)

    def PerformMaintenance(self, request, global_params=None):
        """Perform maintenance on a subset of nodes in the node group.

      Args:
        request: (ComputeNodeGroupsPerformMaintenanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PerformMaintenance')
        return self._RunMethod(config, request, global_params=global_params)
    PerformMaintenance.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.performMaintenance', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}/performMaintenance', request_field='nodeGroupsPerformMaintenanceRequest', request_type_name='ComputeNodeGroupsPerformMaintenanceRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeNodeGroupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.setIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/nodeGroups/{resource}/setIamPolicy', request_field='zoneSetPolicyRequest', request_type_name='ComputeNodeGroupsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetNodeTemplate(self, request, global_params=None):
        """Updates the node template of the node group.

      Args:
        request: (ComputeNodeGroupsSetNodeTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetNodeTemplate')
        return self._RunMethod(config, request, global_params=global_params)
    SetNodeTemplate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.setNodeTemplate', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}/setNodeTemplate', request_field='nodeGroupsSetNodeTemplateRequest', request_type_name='ComputeNodeGroupsSetNodeTemplateRequest', response_type_name='Operation', supports_download=False)

    def SimulateMaintenanceEvent(self, request, global_params=None):
        """Simulates maintenance event on specified nodes from the node group.

      Args:
        request: (ComputeNodeGroupsSimulateMaintenanceEventRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SimulateMaintenanceEvent')
        return self._RunMethod(config, request, global_params=global_params)
    SimulateMaintenanceEvent.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.simulateMaintenanceEvent', ordered_params=['project', 'zone', 'nodeGroup'], path_params=['nodeGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/nodeGroups/{nodeGroup}/simulateMaintenanceEvent', request_field='nodeGroupsSimulateMaintenanceEventRequest', request_type_name='ComputeNodeGroupsSimulateMaintenanceEventRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeNodeGroupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.nodeGroups.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/nodeGroups/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeNodeGroupsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)