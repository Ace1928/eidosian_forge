from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class NamespacesServicesService(base_api.BaseApiService):
    """Service class for the namespaces_services resource."""
    _NAME = 'namespaces_services'

    def __init__(self, client):
        super(RunV1.NamespacesServicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Service. Service creation will trigger a new deployment. Use GetService, and check service.status to determine if the Service is ready.

      Args:
        request: (RunNamespacesServicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/serving.knative.dev/v1/namespaces/{namespacesId}/services', http_method='POST', method_id='run.namespaces.services.create', ordered_params=['parent'], path_params=['parent'], query_params=['dryRun'], relative_path='apis/serving.knative.dev/v1/{+parent}/services', request_field='service', request_type_name='RunNamespacesServicesCreateRequest', response_type_name='Service', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the provided service. This will cause the Service to stop serving traffic and will delete all associated Revisions.

      Args:
        request: (RunNamespacesServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Status) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/serving.knative.dev/v1/namespaces/{namespacesId}/services/{servicesId}', http_method='DELETE', method_id='run.namespaces.services.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'dryRun', 'kind', 'propagationPolicy'], relative_path='apis/serving.knative.dev/v1/{+name}', request_field='', request_type_name='RunNamespacesServicesDeleteRequest', response_type_name='Status', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a service.

      Args:
        request: (RunNamespacesServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/serving.knative.dev/v1/namespaces/{namespacesId}/services/{servicesId}', http_method='GET', method_id='run.namespaces.services.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/serving.knative.dev/v1/{+name}', request_field='', request_type_name='RunNamespacesServicesGetRequest', response_type_name='Service', supports_download=False)

    def List(self, request, global_params=None):
        """Lists services for the given project and region.

      Args:
        request: (RunNamespacesServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/serving.knative.dev/v1/namespaces/{namespacesId}/services', http_method='GET', method_id='run.namespaces.services.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/serving.knative.dev/v1/{+parent}/services', request_field='', request_type_name='RunNamespacesServicesListRequest', response_type_name='ListServicesResponse', supports_download=False)

    def ReplaceService(self, request, global_params=None):
        """Replaces a service. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (RunNamespacesServicesReplaceServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('ReplaceService')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceService.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/serving.knative.dev/v1/namespaces/{namespacesId}/services/{servicesId}', http_method='PUT', method_id='run.namespaces.services.replaceService', ordered_params=['name'], path_params=['name'], query_params=['dryRun'], relative_path='apis/serving.knative.dev/v1/{+name}', request_field='service', request_type_name='RunNamespacesServicesReplaceServiceRequest', response_type_name='Service', supports_download=False)