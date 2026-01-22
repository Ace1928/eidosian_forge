from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class ProjectsLocationsAuditReportsControlReportsFindingsService(base_api.BaseApiService):
    """Service class for the projects_locations_auditReports_controlReports_findings resource."""
    _NAME = 'projects_locations_auditReports_controlReports_findings'

    def __init__(self, client):
        super(AuditmanagerV1alpha.ProjectsLocationsAuditReportsControlReportsFindingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the finding from a control report.

      Args:
        request: (AuditmanagerProjectsLocationsAuditReportsControlReportsFindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Finding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/auditReports/{auditReportsId}/controlReports/{controlReportsId}/findings/{findingsId}', http_method='GET', method_id='auditmanager.projects.locations.auditReports.controlReports.findings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AuditmanagerProjectsLocationsAuditReportsControlReportsFindingsGetRequest', response_type_name='Finding', supports_download=False)

    def List(self, request, global_params=None):
        """Fetches all findings under the control report.

      Args:
        request: (AuditmanagerProjectsLocationsAuditReportsControlReportsFindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/auditReports/{auditReportsId}/controlReports/{controlReportsId}/findings', http_method='GET', method_id='auditmanager.projects.locations.auditReports.controlReports.findings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/findings', request_field='', request_type_name='AuditmanagerProjectsLocationsAuditReportsControlReportsFindingsListRequest', response_type_name='ListFindingsResponse', supports_download=False)