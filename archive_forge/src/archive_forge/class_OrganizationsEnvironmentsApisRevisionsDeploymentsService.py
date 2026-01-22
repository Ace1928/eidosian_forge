from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsApisRevisionsDeploymentsService(base_api.BaseApiService):
    """Service class for the organizations_environments_apis_revisions_deployments resource."""
    _NAME = 'organizations_environments_apis_revisions_deployments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsApisRevisionsDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def GenerateDeployChangeReport(self, request, global_params=None):
        """Generates a report for a dry run analysis of a DeployApiProxy request without committing the deployment. In addition to the standard validations performed when adding deployments, additional analysis will be done to detect possible traffic routing changes that would result from this deployment being created. Any potential routing conflicts or unsafe changes will be reported in the response. This routing analysis is not performed for a non-dry-run DeployApiProxy request. For a request path `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}/deployments:generateDeployChangeReport`, two permissions are required: * `apigee.deployments.create` on the resource `organizations/{org}/environments/{env}` * `apigee.proxyrevisions.deploy` on the resource `organizations/{org}/apis/{api}/revisions/{rev}`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateDeployChangeReportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeploymentChangeReport) The response message.
      """
        config = self.GetMethodConfig('GenerateDeployChangeReport')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateDeployChangeReport.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/deployments:generateDeployChangeReport', http_method='POST', method_id='apigee.organizations.environments.apis.revisions.deployments.generateDeployChangeReport', ordered_params=['name'], path_params=['name'], query_params=['override'], relative_path='v1/{+name}/deployments:generateDeployChangeReport', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateDeployChangeReportRequest', response_type_name='GoogleCloudApigeeV1DeploymentChangeReport', supports_download=False)

    def GenerateUndeployChangeReport(self, request, global_params=None):
        """Generates a report for a dry run analysis of an UndeployApiProxy request without committing the undeploy. In addition to the standard validations performed when removing deployments, additional analysis will be done to detect possible traffic routing changes that would result from this deployment being removed. Any potential routing conflicts or unsafe changes will be reported in the response. This routing analysis is not performed for a non-dry-run UndeployApiProxy request. For a request path `organizations/{org}/environments/{env}/apis/{api}/revisions/{rev}/deployments:generateUndeployChangeReport`, two permissions are required: * `apigee.deployments.delete` on the resource `organizations/{org}/environments/{env}` * `apigee.proxyrevisions.undeploy` on the resource `organizations/{org}/apis/{api}/revisions/{rev}`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateUndeployChangeReportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeploymentChangeReport) The response message.
      """
        config = self.GetMethodConfig('GenerateUndeployChangeReport')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateUndeployChangeReport.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apis/{apisId}/revisions/{revisionsId}/deployments:generateUndeployChangeReport', http_method='POST', method_id='apigee.organizations.environments.apis.revisions.deployments.generateUndeployChangeReport', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/deployments:generateUndeployChangeReport', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsApisRevisionsDeploymentsGenerateUndeployChangeReportRequest', response_type_name='GoogleCloudApigeeV1DeploymentChangeReport', supports_download=False)