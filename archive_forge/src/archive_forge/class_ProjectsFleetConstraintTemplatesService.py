from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsFleetConstraintTemplatesService(base_api.BaseApiService):
    """Service class for the projects_fleetConstraintTemplates resource."""
    _NAME = 'projects_fleetConstraintTemplates'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsFleetConstraintTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves status for a single constraint template. The status is aggregated across all member clusters in a fleet.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FleetConstraintTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/fleetConstraintTemplates/{fleetConstraintTemplatesId}', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.fleetConstraintTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesGetRequest', response_type_name='FleetConstraintTemplate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists aggregate status for constraint templates within a fleet. Each entry in the response contains status for a particular template aggregated across all clusters within a single fleet, in pseudocode the response's shape is: [FleetWideStatusForConstraintTemplate1, FleetWideStatusForConstraintTemplate2, FleetWideStatusForConstraintTemplate3, ...].

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFleetConstraintTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/fleetConstraintTemplates', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.fleetConstraintTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/fleetConstraintTemplates', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsFleetConstraintTemplatesListRequest', response_type_name='ListFleetConstraintTemplatesResponse', supports_download=False)