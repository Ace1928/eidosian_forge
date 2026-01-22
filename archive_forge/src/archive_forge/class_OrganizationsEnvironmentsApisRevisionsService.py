from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsApisRevisionsService(base_api.BaseApiService):
    """Service class for the organizations_environments_apis_revisions resource."""
    _NAME = 'organizations_environments_apis_revisions'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsApisRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Deploy(self, request, global_params=None):
        """Deploys a revision of an API proxy. If another revision of the same API proxy revision is currently deployed, set the `override` parameter to `true` to have this revision replace the currently deployed revision. You cannot invoke an API proxy until it has been deployed to an environment. After you deploy an API proxy revision, you cannot edit it. To edit the API proxy, you must create and deploy a new revision. For a request path `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}/deployments`, two permissions are required: * `apigee.deployments.create` on the resource `organizations/{org}/environments/{env}` * `apigee.proxyrevisions.deploy` on the resource `organizations/{org}/apis/{api}/revisions/{rev}` .

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Deployment) The response message.
      """
        config = self.GetMethodConfig('Deploy')
        return self._RunMethod(config, request, global_params=global_params)
    Deploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/deployments', http_method='POST', method_id='apigee.organizations.environments.apis.revisions.deploy', ordered_params=['name'], path_params=['name'], query_params=['override', 'sequencedRollout', 'serviceAccount'], relative_path='v1/{+name}/deployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDeployRequest', response_type_name='GoogleCloudApigeeV1Deployment', supports_download=False)

    def GetDeployments(self, request, global_params=None):
        """Gets the deployment of an API proxy revision and actual state reported by runtime pods.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsGetDeploymentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Deployment) The response message.
      """
        config = self.GetMethodConfig('GetDeployments')
        return self._RunMethod(config, request, global_params=global_params)
    GetDeployments.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/deployments', http_method='GET', method_id='apigee.organizations.environments.apis.revisions.getDeployments', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/deployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsGetDeploymentsRequest', response_type_name='GoogleCloudApigeeV1Deployment', supports_download=False)

    def Undeploy(self, request, global_params=None):
        """Undeploys an API proxy revision from an environment. For a request path `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}/deployments`, two permissions are required: * `apigee.deployments.delete` on the resource `organizations/{org}/environments/{env}` * `apigee.proxyrevisions.undeploy` on the resource `organizations/{org}/apis/{api}/revisions/{rev}`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsUndeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Undeploy')
        return self._RunMethod(config, request, global_params=global_params)
    Undeploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/deployments', http_method='DELETE', method_id='apigee.organizations.environments.apis.revisions.undeploy', ordered_params=['name'], path_params=['name'], query_params=['sequencedRollout'], relative_path='v1/{+name}/deployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsUndeployRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)