from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class InstanceGroupManagerResizeRequestsService(base_api.BaseApiService):
    """Service class for the instanceGroupManagerResizeRequests resource."""
    _NAME = 'instanceGroupManagerResizeRequests'

    def __init__(self, client):
        super(ComputeBeta.InstanceGroupManagerResizeRequestsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels the specified resize request and removes it from the queue. Cancelled resize request does no longer wait for the resources to be provisioned. Cancel is only possible for requests that are accepted in the queue.

      Args:
        request: (ComputeInstanceGroupManagerResizeRequestsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagerResizeRequests.cancel', ordered_params=['project', 'zone', 'instanceGroupManager', 'resizeRequest'], path_params=['instanceGroupManager', 'project', 'resizeRequest', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resizeRequests/{resizeRequest}/cancel', request_field='', request_type_name='ComputeInstanceGroupManagerResizeRequestsCancelRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified, inactive resize request. Requests that are still active cannot be deleted. Deleting request does not delete instances that were provisioned previously.

      Args:
        request: (ComputeInstanceGroupManagerResizeRequestsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.instanceGroupManagerResizeRequests.delete', ordered_params=['project', 'zone', 'instanceGroupManager', 'resizeRequest'], path_params=['instanceGroupManager', 'project', 'resizeRequest', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resizeRequests/{resizeRequest}', request_field='', request_type_name='ComputeInstanceGroupManagerResizeRequestsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns all of the details about the specified resize request.

      Args:
        request: (ComputeInstanceGroupManagerResizeRequestsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManagerResizeRequest) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceGroupManagerResizeRequests.get', ordered_params=['project', 'zone', 'instanceGroupManager', 'resizeRequest'], path_params=['instanceGroupManager', 'project', 'resizeRequest', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resizeRequests/{resizeRequest}', request_field='', request_type_name='ComputeInstanceGroupManagerResizeRequestsGetRequest', response_type_name='InstanceGroupManagerResizeRequest', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new resize request that starts provisioning VMs immediately or queues VM creation.

      Args:
        request: (ComputeInstanceGroupManagerResizeRequestsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.instanceGroupManagerResizeRequests.insert', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resizeRequests', request_field='instanceGroupManagerResizeRequest', request_type_name='ComputeInstanceGroupManagerResizeRequestsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of resize requests that are contained in the managed instance group.

      Args:
        request: (ComputeInstanceGroupManagerResizeRequestsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceGroupManagerResizeRequestsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.instanceGroupManagerResizeRequests.list', ordered_params=['project', 'zone', 'instanceGroupManager'], path_params=['instanceGroupManager', 'project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/instanceGroupManagers/{instanceGroupManager}/resizeRequests', request_field='', request_type_name='ComputeInstanceGroupManagerResizeRequestsListRequest', response_type_name='InstanceGroupManagerResizeRequestsListResponse', supports_download=False)