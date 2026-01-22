from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsMembershipsService(base_api.BaseApiService):
    """Service class for the projects_memberships resource."""
    _NAME = 'projects_memberships'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsMembershipsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """ListMembershipsProducer returns runtime status from memberships of a fleet.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipsProducerResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/memberships', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.memberships.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/memberships', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipsListRequest', response_type_name='ListMembershipsProducerResponse', supports_download=False)