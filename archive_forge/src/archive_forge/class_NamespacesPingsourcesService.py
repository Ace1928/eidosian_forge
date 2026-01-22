from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1beta1 import anthosevents_v1beta1_messages as messages
class NamespacesPingsourcesService(base_api.BaseApiService):
    """Service class for the namespaces_pingsources resource."""
    _NAME = 'namespaces_pingsources'

    def __init__(self, client):
        super(AnthoseventsV1beta1.NamespacesPingsourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new pingsource.

      Args:
        request: (AnthoseventsNamespacesPingsourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PingSource) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/pingsources', http_method='POST', method_id='anthosevents.namespaces.pingsources.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='apis/sources.knative.dev/v1beta1/{+parent}/pingsources', request_field='pingSource', request_type_name='AnthoseventsNamespacesPingsourcesCreateRequest', response_type_name='PingSource', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a pingsource.

      Args:
        request: (AnthoseventsNamespacesPingsourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/pingsources/{pingsourcesId}', http_method='DELETE', method_id='anthosevents.namespaces.pingsources.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'kind', 'propagationPolicy'], relative_path='apis/sources.knative.dev/v1beta1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesPingsourcesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a pingsource.

      Args:
        request: (AnthoseventsNamespacesPingsourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PingSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/pingsources/{pingsourcesId}', http_method='GET', method_id='anthosevents.namespaces.pingsources.get', ordered_params=['name'], path_params=['name'], query_params=['region'], relative_path='apis/sources.knative.dev/v1beta1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesPingsourcesGetRequest', response_type_name='PingSource', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list pingsources.

      Args:
        request: (AnthoseventsNamespacesPingsourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPingSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/pingsources', http_method='GET', method_id='anthosevents.namespaces.pingsources.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/sources.knative.dev/v1beta1/{+parent}/pingsources', request_field='', request_type_name='AnthoseventsNamespacesPingsourcesListRequest', response_type_name='ListPingSourcesResponse', supports_download=False)

    def ReplacePingSource(self, request, global_params=None):
        """Rpc to replace a pingsource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesPingsourcesReplacePingSourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PingSource) The response message.
      """
        config = self.GetMethodConfig('ReplacePingSource')
        return self._RunMethod(config, request, global_params=global_params)
    ReplacePingSource.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/sources.knative.dev/v1beta1/namespaces/{namespacesId}/pingsources/{pingsourcesId}', http_method='PUT', method_id='anthosevents.namespaces.pingsources.replacePingSource', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/sources.knative.dev/v1beta1/{+name}', request_field='pingSource', request_type_name='AnthoseventsNamespacesPingsourcesReplacePingSourceRequest', response_type_name='PingSource', supports_download=False)