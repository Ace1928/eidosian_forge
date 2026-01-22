from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class SubnetworksService(base_api.BaseApiService):
    """Service class for the subnetworks resource."""
    _NAME = 'subnetworks'

    def __init__(self, client):
        super(ComputeBeta.SubnetworksService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of subnetworks. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeSubnetworksAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SubnetworkAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.subnetworks.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/subnetworks', request_field='', request_type_name='ComputeSubnetworksAggregatedListRequest', response_type_name='SubnetworkAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified subnetwork.

      Args:
        request: (ComputeSubnetworksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.subnetworks.delete', ordered_params=['project', 'region', 'subnetwork'], path_params=['project', 'region', 'subnetwork'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/subnetworks/{subnetwork}', request_field='', request_type_name='ComputeSubnetworksDeleteRequest', response_type_name='Operation', supports_download=False)

    def ExpandIpCidrRange(self, request, global_params=None):
        """Expands the IP CIDR range of the subnetwork to a specified value.

      Args:
        request: (ComputeSubnetworksExpandIpCidrRangeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ExpandIpCidrRange')
        return self._RunMethod(config, request, global_params=global_params)
    ExpandIpCidrRange.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.subnetworks.expandIpCidrRange', ordered_params=['project', 'region', 'subnetwork'], path_params=['project', 'region', 'subnetwork'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/subnetworks/{subnetwork}/expandIpCidrRange', request_field='subnetworksExpandIpCidrRangeRequest', request_type_name='ComputeSubnetworksExpandIpCidrRangeRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified subnetwork.

      Args:
        request: (ComputeSubnetworksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subnetwork) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.subnetworks.get', ordered_params=['project', 'region', 'subnetwork'], path_params=['project', 'region', 'subnetwork'], query_params=[], relative_path='projects/{project}/regions/{region}/subnetworks/{subnetwork}', request_field='', request_type_name='ComputeSubnetworksGetRequest', response_type_name='Subnetwork', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeSubnetworksGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.subnetworks.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/subnetworks/{resource}/getIamPolicy', request_field='', request_type_name='ComputeSubnetworksGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a subnetwork in the specified project using the data included in the request.

      Args:
        request: (ComputeSubnetworksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.subnetworks.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/subnetworks', request_field='subnetwork', request_type_name='ComputeSubnetworksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of subnetworks available to the specified project.

      Args:
        request: (ComputeSubnetworksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SubnetworkList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.subnetworks.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/subnetworks', request_field='', request_type_name='ComputeSubnetworksListRequest', response_type_name='SubnetworkList', supports_download=False)

    def ListUsable(self, request, global_params=None):
        """Retrieves an aggregated list of all usable subnetworks in the project.

      Args:
        request: (ComputeSubnetworksListUsableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UsableSubnetworksAggregatedList) The response message.
      """
        config = self.GetMethodConfig('ListUsable')
        return self._RunMethod(config, request, global_params=global_params)
    ListUsable.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.subnetworks.listUsable', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProject'], relative_path='projects/{project}/aggregated/subnetworks/listUsable', request_field='', request_type_name='ComputeSubnetworksListUsableRequest', response_type_name='UsableSubnetworksAggregatedList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified subnetwork with the data included in the request. Only certain fields can be updated with a patch request as indicated in the field descriptions. You must specify the current fingerprint of the subnetwork resource being patched.

      Args:
        request: (ComputeSubnetworksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.subnetworks.patch', ordered_params=['project', 'region', 'subnetwork'], path_params=['project', 'region', 'subnetwork'], query_params=['drainTimeoutSeconds', 'requestId'], relative_path='projects/{project}/regions/{region}/subnetworks/{subnetwork}', request_field='subnetworkResource', request_type_name='ComputeSubnetworksPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeSubnetworksSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.subnetworks.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/subnetworks/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeSubnetworksSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetPrivateIpGoogleAccess(self, request, global_params=None):
        """Set whether VMs in this subnet can access Google services without assigning external IP addresses through Private Google Access.

      Args:
        request: (ComputeSubnetworksSetPrivateIpGoogleAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetPrivateIpGoogleAccess')
        return self._RunMethod(config, request, global_params=global_params)
    SetPrivateIpGoogleAccess.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.subnetworks.setPrivateIpGoogleAccess', ordered_params=['project', 'region', 'subnetwork'], path_params=['project', 'region', 'subnetwork'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/subnetworks/{subnetwork}/setPrivateIpGoogleAccess', request_field='subnetworksSetPrivateIpGoogleAccessRequest', request_type_name='ComputeSubnetworksSetPrivateIpGoogleAccessRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeSubnetworksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.subnetworks.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/subnetworks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeSubnetworksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)