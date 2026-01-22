from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1beta1 import anthosevents_v1beta1_messages as messages
class NamespacesApiserversourcesService(base_api.BaseApiService):
    """Service class for the namespaces_apiserversources resource."""
    _NAME = 'namespaces_apiserversources'

    def __init__(self, client):
        super(AnthoseventsV1beta1.NamespacesApiserversourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new apiserversource.

      Args:
        request: (AnthoseventsNamespacesApiserversourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiServerSource) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/apiserversources', http_method='POST', method_id='anthosevents.namespaces.apiserversources.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='apis/sources.knative.dev/v1beta1/{+parent}/apiserversources', request_field='apiServerSource', request_type_name='AnthoseventsNamespacesApiserversourcesCreateRequest', response_type_name='ApiServerSource', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a apiserversource.

      Args:
        request: (AnthoseventsNamespacesApiserversourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/apiserversources/{apiserversourcesId}', http_method='DELETE', method_id='anthosevents.namespaces.apiserversources.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'kind', 'propagationPolicy'], relative_path='apis/sources.knative.dev/v1beta1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesApiserversourcesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a apiserversource.

      Args:
        request: (AnthoseventsNamespacesApiserversourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiServerSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/apiserversources/{apiserversourcesId}', http_method='GET', method_id='anthosevents.namespaces.apiserversources.get', ordered_params=['name'], path_params=['name'], query_params=['region'], relative_path='apis/sources.knative.dev/v1beta1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesApiserversourcesGetRequest', response_type_name='ApiServerSource', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list apiserversources.

      Args:
        request: (AnthoseventsNamespacesApiserversourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListApiServerSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/apiserversources', http_method='GET', method_id='anthosevents.namespaces.apiserversources.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/sources.knative.dev/v1beta1/{+parent}/apiserversources', request_field='', request_type_name='AnthoseventsNamespacesApiserversourcesListRequest', response_type_name='ListApiServerSourcesResponse', supports_download=False)

    def ReplaceApiServerSource(self, request, global_params=None):
        """Rpc to replace a apiserversource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesApiserversourcesReplaceApiServerSourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiServerSource) The response message.
      """
        config = self.GetMethodConfig('ReplaceApiServerSource')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceApiServerSource.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/apiserversources/{apiserversourcesId}', http_method='PUT', method_id='anthosevents.namespaces.apiserversources.replaceApiServerSource', ordered_params=['name'], path_params=['name'], query_params=['region'], relative_path='apis/sources.knative.dev/v1beta1/{+name}', request_field='apiServerSource', request_type_name='AnthoseventsNamespacesApiserversourcesReplaceApiServerSourceRequest', response_type_name='ApiServerSource', supports_download=False)