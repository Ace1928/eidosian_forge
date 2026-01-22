from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class GlobalNetworkEndpointGroupsService(base_api.BaseApiService):
    """Service class for the globalNetworkEndpointGroups resource."""
    _NAME = 'globalNetworkEndpointGroups'

    def __init__(self, client):
        super(ComputeBeta.GlobalNetworkEndpointGroupsService, self).__init__(client)
        self._upload_configs = {}

    def AttachNetworkEndpoints(self, request, global_params=None):
        """Attach a network endpoint to the specified network endpoint group.

      Args:
        request: (ComputeGlobalNetworkEndpointGroupsAttachNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AttachNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    AttachNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalNetworkEndpointGroups.attachNetworkEndpoints', ordered_params=['project', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networkEndpointGroups/{networkEndpointGroup}/attachNetworkEndpoints', request_field='globalNetworkEndpointGroupsAttachEndpointsRequest', request_type_name='ComputeGlobalNetworkEndpointGroupsAttachNetworkEndpointsRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified network endpoint group.Note that the NEG cannot be deleted if there are backend services referencing it.

      Args:
        request: (ComputeGlobalNetworkEndpointGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.globalNetworkEndpointGroups.delete', ordered_params=['project', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networkEndpointGroups/{networkEndpointGroup}', request_field='', request_type_name='ComputeGlobalNetworkEndpointGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def DetachNetworkEndpoints(self, request, global_params=None):
        """Detach the network endpoint from the specified network endpoint group.

      Args:
        request: (ComputeGlobalNetworkEndpointGroupsDetachNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DetachNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    DetachNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalNetworkEndpointGroups.detachNetworkEndpoints', ordered_params=['project', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networkEndpointGroups/{networkEndpointGroup}/detachNetworkEndpoints', request_field='globalNetworkEndpointGroupsDetachEndpointsRequest', request_type_name='ComputeGlobalNetworkEndpointGroupsDetachNetworkEndpointsRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified network endpoint group.

      Args:
        request: (ComputeGlobalNetworkEndpointGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalNetworkEndpointGroups.get', ordered_params=['project', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project'], query_params=[], relative_path='projects/{project}/global/networkEndpointGroups/{networkEndpointGroup}', request_field='', request_type_name='ComputeGlobalNetworkEndpointGroupsGetRequest', response_type_name='NetworkEndpointGroup', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a network endpoint group in the specified project using the parameters that are included in the request.

      Args:
        request: (ComputeGlobalNetworkEndpointGroupsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalNetworkEndpointGroups.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/networkEndpointGroups', request_field='networkEndpointGroup', request_type_name='ComputeGlobalNetworkEndpointGroupsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of network endpoint groups that are located in the specified project.

      Args:
        request: (ComputeGlobalNetworkEndpointGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalNetworkEndpointGroups.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/networkEndpointGroups', request_field='', request_type_name='ComputeGlobalNetworkEndpointGroupsListRequest', response_type_name='NetworkEndpointGroupList', supports_download=False)

    def ListNetworkEndpoints(self, request, global_params=None):
        """Lists the network endpoints in the specified network endpoint group.

      Args:
        request: (ComputeGlobalNetworkEndpointGroupsListNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupsListNetworkEndpoints) The response message.
      """
        config = self.GetMethodConfig('ListNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    ListNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalNetworkEndpointGroups.listNetworkEndpoints', ordered_params=['project', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/networkEndpointGroups/{networkEndpointGroup}/listNetworkEndpoints', request_field='', request_type_name='ComputeGlobalNetworkEndpointGroupsListNetworkEndpointsRequest', response_type_name='NetworkEndpointGroupsListNetworkEndpoints', supports_download=False)