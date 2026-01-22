from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSharedflowsRevisionsDeploymentsService(base_api.BaseApiService):
    """Service class for the organizations_sharedflows_revisions_deployments resource."""
    _NAME = 'organizations_sharedflows_revisions_deployments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSharedflowsRevisionsDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all deployments of a shared flow revision.

      Args:
        request: (ApigeeOrganizationsSharedflowsRevisionsDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/sharedflows/{sharedflowsId}/revisions/{revisionsId}/deployments', http_method='GET', method_id='apigee.organizations.sharedflows.revisions.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/deployments', request_field='', request_type_name='ApigeeOrganizationsSharedflowsRevisionsDeploymentsListRequest', response_type_name='GoogleCloudApigeeV1ListDeploymentsResponse', supports_download=False)