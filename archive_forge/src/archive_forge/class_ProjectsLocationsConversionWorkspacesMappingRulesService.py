from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1 import datamigration_v1_messages as messages
class ProjectsLocationsConversionWorkspacesMappingRulesService(base_api.BaseApiService):
    """Service class for the projects_locations_conversionWorkspaces_mappingRules resource."""
    _NAME = 'projects_locations_conversionWorkspaces_mappingRules'

    def __init__(self, client):
        super(DatamigrationV1.ProjectsLocationsConversionWorkspacesMappingRulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new mapping rule for a given conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesMappingRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MappingRule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}/mappingRules', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.mappingRules.create', ordered_params=['parent'], path_params=['parent'], query_params=['mappingRuleId', 'requestId'], relative_path='v1/{+parent}/mappingRules', request_field='mappingRule', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesMappingRulesCreateRequest', response_type_name='MappingRule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single mapping rule.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesMappingRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}/mappingRules/{mappingRulesId}', http_method='DELETE', method_id='datamigration.projects.locations.conversionWorkspaces.mappingRules.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesMappingRulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a mapping rule.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesMappingRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MappingRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}/mappingRules/{mappingRulesId}', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.mappingRules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesMappingRulesGetRequest', response_type_name='MappingRule', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports the mapping rules for a given conversion workspace. Supports various formats of external rules files.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesMappingRulesImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}/mappingRules:import', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.mappingRules.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/mappingRules:import', request_field='importMappingRulesRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesMappingRulesImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the mapping rules for a specific conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesMappingRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMappingRulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}/mappingRules', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.mappingRules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/mappingRules', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesMappingRulesListRequest', response_type_name='ListMappingRulesResponse', supports_download=False)