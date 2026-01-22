from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ProjectsLocationsDomainmappingsService(base_api.BaseApiService):
    """Service class for the projects_locations_domainmappings resource."""
    _NAME = 'projects_locations_domainmappings'

    def __init__(self, client):
        super(RunV1.ProjectsLocationsDomainmappingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new domain mapping.

      Args:
        request: (RunProjectsLocationsDomainmappingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainMapping) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/domainmappings', http_method='POST', method_id='run.projects.locations.domainmappings.create', ordered_params=['parent'], path_params=['parent'], query_params=['dryRun'], relative_path='v1/{+parent}/domainmappings', request_field='domainMapping', request_type_name='RunProjectsLocationsDomainmappingsCreateRequest', response_type_name='DomainMapping', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a domain mapping.

      Args:
        request: (RunProjectsLocationsDomainmappingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Status) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/domainmappings/{domainmappingsId}', http_method='DELETE', method_id='run.projects.locations.domainmappings.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'dryRun', 'kind', 'propagationPolicy'], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsDomainmappingsDeleteRequest', response_type_name='Status', supports_download=False)

    def Get(self, request, global_params=None):
        """Get information about a domain mapping.

      Args:
        request: (RunProjectsLocationsDomainmappingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainMapping) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/domainmappings/{domainmappingsId}', http_method='GET', method_id='run.projects.locations.domainmappings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsDomainmappingsGetRequest', response_type_name='DomainMapping', supports_download=False)

    def List(self, request, global_params=None):
        """List all domain mappings.

      Args:
        request: (RunProjectsLocationsDomainmappingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDomainMappingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/domainmappings', http_method='GET', method_id='run.projects.locations.domainmappings.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='v1/{+parent}/domainmappings', request_field='', request_type_name='RunProjectsLocationsDomainmappingsListRequest', response_type_name='ListDomainMappingsResponse', supports_download=False)