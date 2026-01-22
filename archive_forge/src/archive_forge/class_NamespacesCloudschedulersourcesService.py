from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class NamespacesCloudschedulersourcesService(base_api.BaseApiService):
    """Service class for the namespaces_cloudschedulersources resource."""
    _NAME = 'namespaces_cloudschedulersources'

    def __init__(self, client):
        super(AnthoseventsV1.NamespacesCloudschedulersourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new cloudschedulersource.

      Args:
        request: (AnthoseventsNamespacesCloudschedulersourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudSchedulerSource) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudschedulersources', http_method='POST', method_id='anthosevents.namespaces.cloudschedulersources.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+parent}/cloudschedulersources', request_field='cloudSchedulerSource', request_type_name='AnthoseventsNamespacesCloudschedulersourcesCreateRequest', response_type_name='CloudSchedulerSource', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a cloudschedulersource.

      Args:
        request: (AnthoseventsNamespacesCloudschedulersourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudschedulersources/{cloudschedulersourcesId}', http_method='DELETE', method_id='anthosevents.namespaces.cloudschedulersources.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'kind', 'propagationPolicy'], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudschedulersourcesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a cloudschedulersource.

      Args:
        request: (AnthoseventsNamespacesCloudschedulersourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudSchedulerSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudschedulersources/{cloudschedulersourcesId}', http_method='GET', method_id='anthosevents.namespaces.cloudschedulersources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudschedulersourcesGetRequest', response_type_name='CloudSchedulerSource', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list cloudschedulersources.

      Args:
        request: (AnthoseventsNamespacesCloudschedulersourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCloudSchedulerSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudschedulersources', http_method='GET', method_id='anthosevents.namespaces.cloudschedulersources.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/events.cloud.google.com/v1/{+parent}/cloudschedulersources', request_field='', request_type_name='AnthoseventsNamespacesCloudschedulersourcesListRequest', response_type_name='ListCloudSchedulerSourcesResponse', supports_download=False)

    def ReplaceCloudSchedulerSource(self, request, global_params=None):
        """Rpc to replace a cloudschedulersource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesCloudschedulersourcesReplaceCloudSchedulerSourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudSchedulerSource) The response message.
      """
        config = self.GetMethodConfig('ReplaceCloudSchedulerSource')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceCloudSchedulerSource.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudschedulersources/{cloudschedulersourcesId}', http_method='PUT', method_id='anthosevents.namespaces.cloudschedulersources.replaceCloudSchedulerSource', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='cloudSchedulerSource', request_type_name='AnthoseventsNamespacesCloudschedulersourcesReplaceCloudSchedulerSourceRequest', response_type_name='CloudSchedulerSource', supports_download=False)