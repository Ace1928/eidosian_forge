from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class ProjectsLocationsAuditReportsControlReportsService(base_api.BaseApiService):
    """Service class for the projects_locations_auditReports_controlReports resource."""
    _NAME = 'projects_locations_auditReports_controlReports'

    def __init__(self, client):
        super(AuditmanagerV1alpha.ProjectsLocationsAuditReportsControlReportsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the audit report for a control.

      Args:
        request: (AuditmanagerProjectsLocationsAuditReportsControlReportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ControlReport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/auditReports/{auditReportsId}/controlReports/{controlReportsId}', http_method='GET', method_id='auditmanager.projects.locations.auditReports.controlReports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AuditmanagerProjectsLocationsAuditReportsControlReportsGetRequest', response_type_name='ControlReport', supports_download=False)

    def List(self, request, global_params=None):
        """Fetches all control reports under the parent.

      Args:
        request: (AuditmanagerProjectsLocationsAuditReportsControlReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListControlReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/auditReports/{auditReportsId}/controlReports', http_method='GET', method_id='auditmanager.projects.locations.auditReports.controlReports.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/controlReports', request_field='', request_type_name='AuditmanagerProjectsLocationsAuditReportsControlReportsListRequest', response_type_name='ListControlReportsResponse', supports_download=False)