from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1alpha1 import anthosevents_v1alpha1_messages as messages
class KuberunsService(base_api.BaseApiService):
    """Service class for the kuberuns resource."""
    _NAME = 'kuberuns'

    def __init__(self, client):
        super(AnthoseventsV1alpha1.KuberunsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new KubeRun resource.

      Args:
        request: (AnthoseventsKuberunsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KubeRun) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='anthosevents.kuberuns.create', ordered_params=[], path_params=[], query_params=['parent'], relative_path='apis/operator.run.cloud.google.com/v1alpha1/kuberuns', request_field='kubeRun', request_type_name='AnthoseventsKuberunsCreateRequest', response_type_name='KubeRun', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a KubeRun.

      Args:
        request: (AnthoseventsKuberunsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/kuberuns/{kuberunsId}', http_method='DELETE', method_id='anthosevents.kuberuns.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='', request_type_name='AnthoseventsKuberunsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a KubeRun resource.

      Args:
        request: (AnthoseventsKuberunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KubeRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/kuberuns/{kuberunsId}', http_method='GET', method_id='anthosevents.kuberuns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='', request_type_name='AnthoseventsKuberunsGetRequest', response_type_name='KubeRun', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list KubeRun resources.

      Args:
        request: (AnthoseventsKuberunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListKubeRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='anthosevents.kuberuns.list', ordered_params=[], path_params=[], query_params=['continue_', 'fieldSelector', 'labelSelector', 'limit', 'parent', 'resourceVersion', 'watch'], relative_path='apis/operator.run.cloud.google.com/v1alpha1/kuberuns', request_field='', request_type_name='AnthoseventsKuberunsListRequest', response_type_name='ListKubeRunsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Rpc to update a KubeRun resource.

      Args:
        request: (AnthoseventsKuberunsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KubeRun) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/kuberuns/{kuberunsId}', http_method='PATCH', method_id='anthosevents.kuberuns.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='kubeRun', request_type_name='AnthoseventsKuberunsPatchRequest', response_type_name='KubeRun', supports_download=False)

    def ReplaceKubeRun(self, request, global_params=None):
        """Rpc to replace a KubeRun resource. Only the spec and metadata labels and annotations are modifiable. After the Update request, KubeRun will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsKuberunsReplaceKubeRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KubeRun) The response message.
      """
        config = self.GetMethodConfig('ReplaceKubeRun')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceKubeRun.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/operator.run.cloud.google.com/v1alpha1/kuberuns/{kuberunsId}', http_method='PUT', method_id='anthosevents.kuberuns.replaceKubeRun', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/operator.run.cloud.google.com/v1alpha1/{+name}', request_field='kubeRun', request_type_name='AnthoseventsKuberunsReplaceKubeRunRequest', response_type_name='KubeRun', supports_download=False)