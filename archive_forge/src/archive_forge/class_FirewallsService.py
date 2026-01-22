from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class FirewallsService(base_api.BaseApiService):
    """Service class for the firewalls resource."""
    _NAME = 'firewalls'

    def __init__(self, client):
        super(ComputeBeta.FirewallsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified firewall.

      Args:
        request: (ComputeFirewallsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.firewalls.delete', ordered_params=['project', 'firewall'], path_params=['firewall', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/firewalls/{firewall}', request_field='', request_type_name='ComputeFirewallsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified firewall.

      Args:
        request: (ComputeFirewallsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Firewall) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.firewalls.get', ordered_params=['project', 'firewall'], path_params=['firewall', 'project'], query_params=[], relative_path='projects/{project}/global/firewalls/{firewall}', request_field='', request_type_name='ComputeFirewallsGetRequest', response_type_name='Firewall', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a firewall rule in the specified project using the data included in the request.

      Args:
        request: (ComputeFirewallsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.firewalls.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/firewalls', request_field='firewall', request_type_name='ComputeFirewallsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of firewall rules available to the specified project.

      Args:
        request: (ComputeFirewallsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.firewalls.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/firewalls', request_field='', request_type_name='ComputeFirewallsListRequest', response_type_name='FirewallList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified firewall rule with the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeFirewallsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.firewalls.patch', ordered_params=['project', 'firewall'], path_params=['firewall', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/firewalls/{firewall}', request_field='firewallResource', request_type_name='ComputeFirewallsPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeFirewallsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.firewalls.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/firewalls/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeFirewallsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified firewall rule with the data included in the request. Note that all fields will be updated if using PUT, even fields that are not specified. To update individual fields, please use PATCH instead.

      Args:
        request: (ComputeFirewallsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.firewalls.update', ordered_params=['project', 'firewall'], path_params=['firewall', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/firewalls/{firewall}', request_field='firewallResource', request_type_name='ComputeFirewallsUpdateRequest', response_type_name='Operation', supports_download=False)