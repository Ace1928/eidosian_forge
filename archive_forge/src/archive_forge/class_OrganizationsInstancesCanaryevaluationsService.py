from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsInstancesCanaryevaluationsService(base_api.BaseApiService):
    """Service class for the organizations_instances_canaryevaluations resource."""
    _NAME = 'organizations_instances_canaryevaluations'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsInstancesCanaryevaluationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new canary evaluation for an organization.

      Args:
        request: (ApigeeOrganizationsInstancesCanaryevaluationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/canaryevaluations', http_method='POST', method_id='apigee.organizations.instances.canaryevaluations.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/canaryevaluations', request_field='googleCloudApigeeV1CanaryEvaluation', request_type_name='ApigeeOrganizationsInstancesCanaryevaluationsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a CanaryEvaluation for an organization.

      Args:
        request: (ApigeeOrganizationsInstancesCanaryevaluationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1CanaryEvaluation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/instances/{instancesId}/canaryevaluations/{canaryevaluationsId}', http_method='GET', method_id='apigee.organizations.instances.canaryevaluations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsInstancesCanaryevaluationsGetRequest', response_type_name='GoogleCloudApigeeV1CanaryEvaluation', supports_download=False)