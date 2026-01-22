from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionInstanceGroupsService(base_api.BaseApiService):
    """Service class for the regionInstanceGroups resource."""
    _NAME = 'regionInstanceGroups'

    def __init__(self, client):
        super(ComputeBeta.RegionInstanceGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the specified instance group resource.

      Args:
        request: (ComputeRegionInstanceGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionInstanceGroups.get', ordered_params=['project', 'region', 'instanceGroup'], path_params=['instanceGroup', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/instanceGroups/{instanceGroup}', request_field='', request_type_name='ComputeRegionInstanceGroupsGetRequest', response_type_name='InstanceGroup', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of instance group resources contained within the specified region.

      Args:
        request: (ComputeRegionInstanceGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionInstanceGroups.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/instanceGroups', request_field='', request_type_name='ComputeRegionInstanceGroupsListRequest', response_type_name='RegionInstanceGroupList', supports_download=False)

    def ListInstances(self, request, global_params=None):
        """Lists the instances in the specified instance group and displays information about the named ports. Depending on the specified options, this method can list all instances or only the instances that are running. The orderBy query parameter is not supported.

      Args:
        request: (ComputeRegionInstanceGroupsListInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupsListInstances) The response message.
      """
        config = self.GetMethodConfig('ListInstances')
        return self._RunMethod(config, request, global_params=global_params)
    ListInstances.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroups.listInstances', ordered_params=['project', 'region', 'instanceGroup'], path_params=['instanceGroup', 'project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/instanceGroups/{instanceGroup}/listInstances', request_field='regionInstanceGroupsListInstancesRequest', request_type_name='ComputeRegionInstanceGroupsListInstancesRequest', response_type_name='RegionInstanceGroupsListInstances', supports_download=False)

    def SetNamedPorts(self, request, global_params=None):
        """Sets the named ports for the specified regional instance group.

      Args:
        request: (ComputeRegionInstanceGroupsSetNamedPortsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetNamedPorts')
        return self._RunMethod(config, request, global_params=global_params)
    SetNamedPorts.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroups.setNamedPorts', ordered_params=['project', 'region', 'instanceGroup'], path_params=['instanceGroup', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceGroups/{instanceGroup}/setNamedPorts', request_field='regionInstanceGroupsSetNamedPortsRequest', request_type_name='ComputeRegionInstanceGroupsSetNamedPortsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionInstanceGroupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceGroups.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/instanceGroups/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionInstanceGroupsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)