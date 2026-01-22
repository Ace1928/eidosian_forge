from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsSimulationsAttackExposureResultsAttackPathsService(base_api.BaseApiService):
    """Service class for the organizations_simulations_attackExposureResults_attackPaths resource."""
    _NAME = 'organizations_simulations_attackExposureResults_attackPaths'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsSimulationsAttackExposureResultsAttackPathsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the attack paths for a set of simulation results or valued resources and filter.

      Args:
        request: (SecuritycenterOrganizationsSimulationsAttackExposureResultsAttackPathsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAttackPathsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/simulations/{simulationsId}/attackExposureResults/{attackExposureResultsId}/attackPaths', http_method='GET', method_id='securitycenter.organizations.simulations.attackExposureResults.attackPaths.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/attackPaths', request_field='', request_type_name='SecuritycenterOrganizationsSimulationsAttackExposureResultsAttackPathsListRequest', response_type_name='ListAttackPathsResponse', supports_download=False)