from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class ProjectsLocationsEntryGroupsTagsService(base_api.BaseApiService):
    """Service class for the projects_locations_entryGroups_tags resource."""
    _NAME = 'projects_locations_entryGroups_tags'

    def __init__(self, client):
        super(DatacatalogV1.ProjectsLocationsEntryGroupsTagsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a tag and assigns it to: * An Entry if the method name is `projects.locations.entryGroups.entries.tags.create`. * Or EntryGroupif the method name is `projects.locations.entryGroups.tags.create`. Note: The project identified by the `parent` parameter for the [tag] (https://cloud.google.com/data-catalog/docs/reference/rest/v1/projects.locations.entryGroups.entries.tags/create#path-parameters) and the [tag template] (https://cloud.google.com/data-catalog/docs/reference/rest/v1/projects.locations.tagTemplates/create#path-parameters) used to create the tag must be in the same organization.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsTagsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Tag) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/tags', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.tags.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/tags', request_field='googleCloudDatacatalogV1Tag', request_type_name='DatacatalogProjectsLocationsEntryGroupsTagsCreateRequest', response_type_name='GoogleCloudDatacatalogV1Tag', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a tag.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsTagsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/tags/{tagsId}', http_method='DELETE', method_id='datacatalog.projects.locations.entryGroups.tags.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsTagsDeleteRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """Lists tags assigned to an Entry. The columns in the response are lowercased.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsTagsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1ListTagsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/tags', http_method='GET', method_id='datacatalog.projects.locations.entryGroups.tags.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/tags', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsTagsListRequest', response_type_name='GoogleCloudDatacatalogV1ListTagsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing tag.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsTagsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Tag) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/tags/{tagsId}', http_method='PATCH', method_id='datacatalog.projects.locations.entryGroups.tags.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudDatacatalogV1Tag', request_type_name='DatacatalogProjectsLocationsEntryGroupsTagsPatchRequest', response_type_name='GoogleCloudDatacatalogV1Tag', supports_download=False)