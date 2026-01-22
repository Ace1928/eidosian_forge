from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1alpha1 import anthosevents_v1alpha1_messages as messages
class NamespacesCloudrunsService(base_api.BaseApiService):
    """Service class for the namespaces_cloudruns resource."""
    _NAME = 'namespaces_cloudruns'

    def __init__(self, client):
        super(AnthoseventsV1alpha1.NamespacesCloudrunsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new CloudRun resource.

      Args:
        request: (AnthoseventsNamespacesCloudrunsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudRun) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/namespaces/{namespacesId}/cloudruns', http_method='POST', method_id='anthosevents.namespaces.cloudruns.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+parent}/cloudruns', request_field='cloudRun', request_type_name='AnthoseventsNamespacesCloudrunsCreateRequest', response_type_name='CloudRun', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a CloudRun.

      Args:
        request: (AnthoseventsNamespacesCloudrunsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/namespaces/{namespacesId}/cloudruns/{cloudrunsId}', http_method='DELETE', method_id='anthosevents.namespaces.cloudruns.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudrunsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a CloudRun resource.

      Args:
        request: (AnthoseventsNamespacesCloudrunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/namespaces/{namespacesId}/cloudruns/{cloudrunsId}', http_method='GET', method_id='anthosevents.namespaces.cloudruns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudrunsGetRequest', response_type_name='CloudRun', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list CloudRun resources.

      Args:
        request: (AnthoseventsNamespacesCloudrunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCloudRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/namespaces/{namespacesId}/cloudruns', http_method='GET', method_id='anthosevents.namespaces.cloudruns.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+parent}/cloudruns', request_field='', request_type_name='AnthoseventsNamespacesCloudrunsListRequest', response_type_name='ListCloudRunsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Rpc to update a CloudRun resource.

      Args:
        request: (AnthoseventsNamespacesCloudrunsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudRun) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/namespaces/{namespacesId}/cloudruns/{cloudrunsId}', http_method='PATCH', method_id='anthosevents.namespaces.cloudruns.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='cloudRun', request_type_name='AnthoseventsNamespacesCloudrunsPatchRequest', response_type_name='CloudRun', supports_download=False)

    def ReplaceCloudRun(self, request, global_params=None):
        """Rpc to replace a CloudRun resource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesCloudrunsReplaceCloudRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudRun) The response message.
      """
        config = self.GetMethodConfig('ReplaceCloudRun')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceCloudRun.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/namespaces/{namespacesId}/cloudruns/{cloudrunsId}', http_method='PUT', method_id='anthosevents.namespaces.cloudruns.replaceCloudRun', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='cloudRun', request_type_name='AnthoseventsNamespacesCloudrunsReplaceCloudRunRequest', response_type_name='CloudRun', supports_download=False)