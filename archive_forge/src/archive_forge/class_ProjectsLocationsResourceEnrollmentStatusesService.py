from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class ProjectsLocationsResourceEnrollmentStatusesService(base_api.BaseApiService):
    """Service class for the projects_locations_resourceEnrollmentStatuses resource."""
    _NAME = 'projects_locations_resourceEnrollmentStatuses'

    def __init__(self, client):
        super(AuditmanagerV1alpha.ProjectsLocationsResourceEnrollmentStatusesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get a resource along with its enrollment status.

      Args:
        request: (AuditmanagerProjectsLocationsResourceEnrollmentStatusesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceEnrollmentStatus) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceEnrollmentStatuses/{resourceEnrollmentStatusesId}', http_method='GET', method_id='auditmanager.projects.locations.resourceEnrollmentStatuses.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AuditmanagerProjectsLocationsResourceEnrollmentStatusesGetRequest', response_type_name='ResourceEnrollmentStatus', supports_download=False)

    def List(self, request, global_params=None):
        """Fetches all resources under the parent along with their enrollment.

      Args:
        request: (AuditmanagerProjectsLocationsResourceEnrollmentStatusesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResourceEnrollmentStatusesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceEnrollmentStatuses', http_method='GET', method_id='auditmanager.projects.locations.resourceEnrollmentStatuses.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/resourceEnrollmentStatuses', request_field='', request_type_name='AuditmanagerProjectsLocationsResourceEnrollmentStatusesListRequest', response_type_name='ListResourceEnrollmentStatusesResponse', supports_download=False)