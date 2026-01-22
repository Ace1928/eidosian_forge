from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsAnnouncementsService(base_api.BaseApiService):
    """Service class for the projects_locations_announcements resource."""
    _NAME = 'projects_locations_announcements'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsAnnouncementsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a `Announcement` by its resource name.

      Args:
        request: (VmwareengineProjectsLocationsAnnouncementsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Announcement) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/announcements/{announcementsId}', http_method='GET', method_id='vmwareengine.projects.locations.announcements.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsAnnouncementsGetRequest', response_type_name='Announcement', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `Announcements` for a given region and project.

      Args:
        request: (VmwareengineProjectsLocationsAnnouncementsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAnnouncementsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/announcements', http_method='GET', method_id='vmwareengine.projects.locations.announcements.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/announcements', request_field='', request_type_name='VmwareengineProjectsLocationsAnnouncementsListRequest', response_type_name='ListAnnouncementsResponse', supports_download=False)