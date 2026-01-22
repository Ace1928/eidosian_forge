from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsFleetConstraintsService(base_api.BaseApiService):
    """Service class for the projects_fleetConstraints resource."""
    _NAME = 'projects_fleetConstraints'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsFleetConstraintsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves fleet-wide constraint info.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsFleetConstraintsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FleetConstraint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/fleetConstraints/{fleetConstraintsId}/{fleetConstraintsId1}', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.fleetConstraints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsFleetConstraintsGetRequest', response_type_name='FleetConstraint', supports_download=False)

    def List(self, request, global_params=None):
        """ListFleetConstraints returns fleet-wide constraint info.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsFleetConstraintsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFleetConstraintsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/fleetConstraints', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.fleetConstraints.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/fleetConstraints', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsFleetConstraintsListRequest', response_type_name='ListFleetConstraintsResponse', supports_download=False)