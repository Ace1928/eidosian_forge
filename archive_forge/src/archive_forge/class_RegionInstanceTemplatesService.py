from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionInstanceTemplatesService(base_api.BaseApiService):
    """Service class for the regionInstanceTemplates resource."""
    _NAME = 'regionInstanceTemplates'

    def __init__(self, client):
        super(ComputeBeta.RegionInstanceTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified instance template. Deleting an instance template is permanent and cannot be undone.

      Args:
        request: (ComputeRegionInstanceTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionInstanceTemplates.delete', ordered_params=['project', 'region', 'instanceTemplate'], path_params=['instanceTemplate', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceTemplates/{instanceTemplate}', request_field='', request_type_name='ComputeRegionInstanceTemplatesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified instance template.

      Args:
        request: (ComputeRegionInstanceTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionInstanceTemplates.get', ordered_params=['project', 'region', 'instanceTemplate'], path_params=['instanceTemplate', 'project', 'region'], query_params=['view'], relative_path='projects/{project}/regions/{region}/instanceTemplates/{instanceTemplate}', request_field='', request_type_name='ComputeRegionInstanceTemplatesGetRequest', response_type_name='InstanceTemplate', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an instance template in the specified project and region using the global instance template whose URL is included in the request.

      Args:
        request: (ComputeRegionInstanceTemplatesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstanceTemplates.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instanceTemplates', request_field='instanceTemplate', request_type_name='ComputeRegionInstanceTemplatesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of instance templates that are contained within the specified project and region.

      Args:
        request: (ComputeRegionInstanceTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceTemplateList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionInstanceTemplates.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'view'], relative_path='projects/{project}/regions/{region}/instanceTemplates', request_field='', request_type_name='ComputeRegionInstanceTemplatesListRequest', response_type_name='InstanceTemplateList', supports_download=False)